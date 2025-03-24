from lib.models import get_net,get_net2, get_net_kp_seg, get_net_kpline
from lib.config import cfg
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/center_threeHead_radio_3_lineEXdata_offset1x_8cat_anchor9_KP_line_finute/Chart2019/_2023-08-30-10-51/epoch-19.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='/root/data1/YOLOP/ChartOCR512x512/image/val/', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    # with torch.no_grad():
    #     detect(cfg,opt)
    model = get_net2(cfg)
    device = 'cuda'
    # model = get_net_kpline(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    checkpoint_dict = {key.replace('module.model.', 'model.'):value for key,value in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint_dict)
    
    model = model.to(device)
    model.eval()

    # 创建一个输入张量
    dummy_input = torch.randn((1,3, 512,512)).to(device)

    # 指定导出的文件路径
    onnx_model_path = "chartparsing_curve_keypoint.onnx"

    # 导出模型
    torch.onnx.export(
        model,                    # 要导出的模型
        dummy_input,              # 模型的一个示例输入
        onnx_model_path,          # 导出的文件路径
        export_params=True,       # 导出模型参数
        opset_version=11,         # ONNX 版本
        do_constant_folding=False, # 是否执行常量折叠优化
        input_names=['input'],    # 输入节点的名称
        output_names=['output'],  # 输出节点的名称
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # 动态轴
    )

    print(f"Model has been converted to ONNX and saved at {onnx_model_path}")