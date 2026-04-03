import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- 配置区 ---
engine_path = '../pretrained/depth_vits.engine' 
input_image_path = 'test.jpg'
input_size = (728, 728)

# 1. 加载 Engine
logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)
output_shape = engine.get_tensor_shape(output_name)

# 分配 GPU 显存
d_input = cuda.mem_alloc(1 * 3 * 728 * 728 * 4) # float32 size
d_output = cuda.mem_alloc(int(np.prod(output_shape)) * 4)
stream = cuda.Stream()

# 准备基础数据
img_pil = Image.open(input_image_path).convert('RGB')
orig_w, orig_h = img_pil.size
img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float().unsqueeze(0)

def run_inference(input_tensor):
    """通用推理函数"""
    # 归一化
    input_tensor = (input_tensor / 255.0 - 0.5) / 0.5
    input_numpy = np.ascontiguousarray(input_tensor.numpy().astype(np.float32))
    
    # 推理过程
    cuda.memcpy_htod_async(d_input, input_numpy, stream)
    context.set_tensor_address(input_name, d_input)
    context.set_tensor_address(output_name, d_output)
    context.execute_async_v3(stream_handle=stream.handle)
    
    h_output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return torch.from_numpy(h_output)

# --- 实验 1: 直接缩放 (Distorted) ---
print("🚀 正在生成：直接缩放版本...")
img_distorted = F.interpolate(img_tensor, size=input_size, mode='bilinear')
out_distorted = run_inference(img_distorted)
# 拉回原图尺寸
depth_distorted = F.interpolate(out_distorted.view(1, 1, 728, 728), size=(orig_h, orig_w), mode='bilinear').squeeze().numpy()

# --- 实验 2: 填充缩放 (Letterbox) ---
print("🚀 正在生成：填充 (Letterbox) 版本...")
scale = 728 / max(orig_h, orig_w)
new_h, new_w = int(orig_h * scale), int(orig_w * scale)
img_scaled = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear')
# 补齐到 728x728
img_letterbox = F.pad(img_scaled, (0, 728 - new_w, 0, 728 - new_h), value=0)
out_letterbox_raw = run_inference(img_letterbox)
# 裁剪有效区域并拉回原图
out_letterbox_valid = out_letterbox_raw.view(1, 1, 728, 728)[:, :, :new_h, :new_w]
depth_letterbox = F.interpolate(out_letterbox_valid, size=(orig_h, orig_w), mode='bilinear').squeeze().numpy()

# --- 后处理与对比保存 ---
def normalize(d):
    d_min, d_max = d.min(), d.max()
    return ((d - d_min) / (d_max - d_min + 1e-5) * 255).astype(np.uint8)

plt.imsave('depth_distorted.png', normalize(depth_distorted), cmap='plasma')
plt.imsave('depth_letterbox.png', normalize(depth_letterbox), cmap='plasma')

print("\n✨ 对比完成！")
print(f"1. 直接缩放结果: depth_distorted.png (物体比例可能变形)")
print(f"2. 填充填充结果: depth_letterbox.png (物体比例保持真实)")