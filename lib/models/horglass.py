import numpy as np
import torch
import torch.nn as nn


#-------------------------#
#   卷积+标准化+激活函数
#-------------------------#
class conv2d(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(conv2d, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

#-------------------------#
#   残差结构
#-------------------------#
class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

def make_layer(k, inp_dim, out_dim, modules, **kwargs):
    layers = [residual(k, inp_dim, out_dim, **kwargs)]
    for _ in range(modules - 1):
        layers.append(residual(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_hg_layer(k, inp_dim, out_dim, modules, **kwargs):
    layers  = [residual(k, inp_dim, out_dim, stride=2)]
    for _ in range(modules - 1):
        layers += [residual(k, out_dim, out_dim)]
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(residual(k, inp_dim, inp_dim, **kwargs))
    layers.append(residual(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


class kp_module(nn.Module):
    def __init__(self, n, dims, modules, **kwargs):
        super(kp_module, self).__init__()
        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        # 将输入进来的特征层进行两次残差卷积，便于和后面的层进行融合
        self.up1  = make_layer(
            3, curr_dim, curr_dim, curr_mod, **kwargs
        )  

        # 进行下采样
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod, **kwargs
        )

        # 构建U形结构的下一层
        if self.n > 1 :
            self.low2 = kp_module(
                n - 1, dims[1:], modules[1:], **kwargs
            ) 
        else:
            self.low2 = make_layer(
                3, next_dim, next_dim, next_mod, **kwargs
            )

        # 将U形结构下一层反馈上来的层进行残差卷积
        self.low3 = make_layer_revr(
            3, next_dim, curr_dim, curr_mod, **kwargs
        )
        # 将U形结构下一层反馈上来的层进行上采样
        self.up2  = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1  = self.up1(x)   # x
        low1 = self.low1(x)  # x/2
        low2 = self.low2(low1) # x/4 
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        outputs = up1 + up2
        return outputs

class PivotDetectin(nn.Module):
    def __init__(self, nclass, pretrained=False, num_stacks=3, n=5):
        super(PivotDetectin, self).__init__()
        if pretrained:
            raise ValueError("HourglassNet has no pretrained model")
        cnv_dim=128
        dims=[128, 128, 192, 192, 192, 256] 
        modules = [2, 2, 2, 2, 2, 4]

        # dims    = [256, 256, 384, 384, 384, 512]
        # modules = [2, 2, 2, 2, 2, 4]
        heads={'hm': nclass,  'reg':2}
        self.nstack    = num_stacks
        self.heads     = heads

        curr_dim = dims[0]

        # self.pre = nn.Sequential(
        #             conv2d(7, 3, 128, stride=2),
        #             residual(3, 128, 128, stride=2)
        #         ) 
        
        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules
            ) for _ in range(num_stacks)
        ])

        self.cnvs = nn.ModuleList([
            conv2d(3, curr_dim, cnv_dim) for _ in range(num_stacks)
        ])

        self.inters = nn.ModuleList([
            residual(3, curr_dim, curr_dim) for _ in range(num_stacks - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])
        
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])

        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    nn.Sequential(
                        conv2d(3, cnv_dim, curr_dim, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], (1, 1))
                    )  for _ in range(num_stacks)

                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    # heat[-1].weight.data.fill_(0)
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    nn.Sequential(
                        conv2d(3, cnv_dim, curr_dim, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], (1, 1))
                    )  for _ in range(num_stacks)
                ])
                self.__setattr__(head, module)


        self.relu = nn.ReLU(inplace=True)

    # def freeze_backbone(self):
    #     freeze_list = [self.pre, self.kps]
    #     for module in freeze_list:
    #         for param in module.parameters():
    #             param.requires_grad = False

    # def unfreeze_backbone(self):
    #     freeze_list = [self.pre, self.kps]
    #     for module in freeze_list:
    #         for param in module.parameters():
    #             param.requires_grad = True

    def forward(self, image):
        # print('image shape', image.shape)
        # inter = self.pre(image)
        inter = image
        outs  = []

        for ind in range(self.nstack):
            kp  = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp)

        
            out = {}
            for head in self.heads:
                out[head] = self.__getattr__(head)[ind](cnv)
                if head=='hm':
                    out[head] = out[head]
                    # if self.heads[head]==1:
                    #     out[head] = out[head].sigmoid()
                    # else:
                    #     out[head] = out[head].softmax(dim=1)
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
            outs.append(out)
        return outs


class Heatmap(nn.Module):
    def __init__(self, nclass, pretrain=False):
        super(Heatmap, self).__init__()
        self.nclass = nclass
        self.conv = nn.Sequential(
                        conv2d(3, 128, 256, with_bn=True),

                       conv2d(3, 256, 512, with_bn=True),
                    )
        
        self.hm = nn.Sequential(
                        conv2d(3, 512, 256, with_bn=True),
                        nn.Conv2d(256, self.nclass, (1, 1)))
        self.reg =  nn.Sequential(
                        conv2d(3, 512, 256, with_bn=True),
                        nn.Conv2d(256, 2, (1, 1))
                         )

    def forward(self, image):
        out = {}
        image = self.conv(image)
        hm = self.hm(image)
        reg = self.reg(image)
        if self.nclass:
            out['hm'] = hm.sigmoid()
        else:
            out['hm'] = hm.softmax(dim=1)
        out['reg'] = reg
        return [out]


# from lib.models.utils_hourglass import *




if __name__=='__main__':

    from thop import profile
    from thop import clever_format
    from utils_hourglass import *
    from torchsummaryX  import summary
    # model2 = hourglass_block(128, 128)


    model = PivotDetectin(3, pretrained = False)
    x = torch.rand(1,128,64,64)
    summary(model,x)
    # macs, params = profile(model, inputs=(x, ))
    # print(macs, params)
    # # print('flops is %.2f G' % (flops/np.power(1024, 3)))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('macs:{}\nparams:{}'.format(macs, params))
    # print('params:{}'.format(clever_format(sum(p.numel() for p in model2.parameters() if p.requires_grad)*4,"%.3f")))
    out = model(x)
    # print(out.shape)
    # import graphviz
    # from torchviz import make_dot
    x = torch.randn(20, 20)
    # print(dict(list(model.named_parameters())))
    # vise=make_dot(out, params=dict(model.named_parameters()))
    # vise.view()

    # import hiddenlayer as h
    # vis_graph = h.build_graph(model, torch.zeros([1 ,128, 64, 64]))   # 获取绘制图像的对象
    # vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
    # vis_graph.save("./demo1.png")   # 保存图像的路径


