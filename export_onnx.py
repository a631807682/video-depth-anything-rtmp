import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))

from depth_anything_v2.dpt import DepthAnythingV2
from torch.nn.attention import SDPBackend, sdpa_kernel

# 1. 初始化并加载模型
model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
model = DepthAnythingV2(**model_configs)
model.load_state_dict(torch.load('pretrained/depth_anything_v2_vits.pth', map_location='cpu'))
model.eval().cuda()

# 2. 准备虚拟输入 (必须是 14 的倍数)
dummy_input = torch.randn(1, 3, 518, 518).cuda()

# 3. 使用上下文管理器禁用加速内核进行导出
# 这是解决 _efficient_attention_forward 报错的关键！
with torch.no_grad():
    with sdpa_kernel(SDPBackend.MATH):
        torch.onnx.export(
            model, 
            dummy_input, 
            "depth_vits_518.onnx",
            opset_version=17,
            input_names=['input'],
            output_names=['output'],
            do_constant_folding=True
        )

print("✅ ONNX 导出成功：depth_anything_v2_vits_518.onnx")
