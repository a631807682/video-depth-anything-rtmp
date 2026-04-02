import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# 1. 读取图片
img = Image.open('test.jpg').convert('RGB')
orig_w, orig_h = img.size

# 2. 预处理
input_size = (728, 728)  # 根据你的模型实际输入尺寸调整
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]

# 3. 转 numpy
img_numpy = img_tensor.numpy().astype(np.float32)

# 4. ONNX 推理
onnx_path = '../pretrained/depth_anything_v2_vits.onnx'  # 替换为你的模型路径
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name
depth = ort_session.run(None, {input_name: img_numpy})[0]  # [1, 1, H, W]
depth_2d = depth[0, 0]  # [H, W]
if depth.ndim == 4:
    # [1, 1, H, W]
    depth_2d = depth[0, 0]
elif depth.ndim == 3:
    # [1, H, W]
    depth_2d = depth[0]
elif depth.ndim == 2:
    # [H, W]
    depth_2d = depth
elif depth.ndim == 1:
    # [H]，错误，模型输出不对
    raise ValueError("模型输出 shape 异常，应该是 [H, W]，实际是: {}".format(depth.shape))
else:
    raise ValueError("未知的 depth shape: {}".format(depth.shape))

depth_tensor = torch.from_numpy(depth_2d).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
depth_resized = torch.nn.functional.interpolate(
    depth_tensor,
    size=(orig_h, orig_w),
    mode='bilinear',
    align_corners=False
).squeeze().numpy()

# 6. 归一化到0-255
d_min, d_max = depth_resized.min(), depth_resized.max()
if d_max - d_min > 1e-5:
    depth_norm = (depth_resized - d_min) / (d_max - d_min)
else:
    depth_norm = np.zeros_like(depth_resized)
depth_uint8 = (depth_norm * 255).astype(np.uint8)

# 7. 保存深度图
plt.imsave('depth_onnx.png', depth_uint8, cmap='plasma')
print('已保存 depth_onnx.png')
