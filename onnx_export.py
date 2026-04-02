import torch
import os
import sys
from unittest.mock import patch

# 1. 核心修复：彻底屏蔽 xformers 模块，防止它干扰导出
sys.modules['xformers'] = None
sys.modules['xformers.ops'] = None

# 2. 设置路径并导入模型
sys.path.append(os.path.join(os.getcwd(), 'Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2

# 3. 定义最基础的数学 Attention (确保 ONNX 100% 兼容)
def manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    scale_factor = 1 / (query.size(-1)**0.5) if scale is None else scale
    # Q @ K.T -> Softmax -> @ V
    attn_weight = (query @ key.transpose(-2, -1) * scale_factor).softmax(dim=-1)
    return attn_weight @ value

# 4. 执行导出
def do_export():
    device = 'cpu' # CPU 导出最稳，避开 CUDA 驱动层的算子分发
    onnx_path = './pretrained/depth_anything_v2_vits.onnx'
    
    # 初始化模型
    model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    model = DepthAnythingV2(**model_configs).to(device)
    model.load_state_dict(torch.load('./pretrained/depth_anything_v2_vits.pth', map_location=device))
    model.eval()

    # 使用 mock 强制拦截所有的 Attention 调用
    print("🚀 正在屏蔽 xformers 并强制使用基础数学 Attention 导出...")
    # 修改导出脚本的关键部分
    with patch('torch.nn.functional.scaled_dot_product_attention', manual_sdpa):
        dummy_input = torch.randn(1, 3, 728, 728).to(device)
        with torch.no_grad():
            torch.onnx.export(
                model, 
                dummy_input, 
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )

    print(f"✅ 终于导出成功: {onnx_path}")

if __name__ == "__main__":
    do_export()
