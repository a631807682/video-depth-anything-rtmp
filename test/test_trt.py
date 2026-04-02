import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 配置区 ---
engine_path = '../pretrained/depth_vits.engine' 
input_image_path = 'test.jpg'
output_image_path = 'depth_engine.png'
input_size = (728, 728)

# 1. 加载 Engine
logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# 2. 预处理
img = Image.open(input_image_path).convert('RGB')
orig_w, orig_h = img.size
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img_tensor = transform(img).unsqueeze(0)
img_numpy = np.ascontiguousarray(img_tensor.numpy().astype(np.float32))

# 3. 获取输入输出张量名称 (TensorRT 10+ 推荐方式)
input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

# 4. 分配显存
h_input = img_numpy
# 自动获取输出 shape
output_shape = engine.get_tensor_shape(output_name)
h_output = np.empty(output_shape, dtype=np.float32)

d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

# 5. 推理 (使用新的 Tensor Address 绑定方式)
# 将数据拷贝到 GPU
cuda.memcpy_htod_async(d_input, h_input, stream)

# 绑定张量地址
context.set_tensor_address(input_name, d_input)
context.set_tensor_address(output_name, d_output)

# 执行异步推理
context.execute_async_v3(stream_handle=stream.handle)

# 拷贝回 CPU
cuda.memcpy_dtoh_async(h_output, d_output, stream)
stream.synchronize()

# 6. 后处理
depth_2d = h_output.squeeze()
depth_tensor = torch.from_numpy(depth_2d).unsqueeze(0).unsqueeze(0)
depth_resized = torch.nn.functional.interpolate(
    depth_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False
).squeeze().numpy()

# 归一化并保存
d_min, d_max = depth_resized.min(), depth_resized.max()
depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-5)
depth_uint8 = (depth_norm * 255).astype(np.uint8)

plt.imsave(output_image_path, depth_uint8, cmap='plasma')
print(f'成功！深度图已保存至: {output_image_path}')

