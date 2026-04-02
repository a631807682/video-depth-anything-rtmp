import tensorrt as trt
import torch
import cupy as cp
import cv2
import numpy as np
from cupyx.scipy.ndimage import map_coordinates

import tensorrt as trt
import torch
import cv2
import numpy as np
import torch.nn.functional as F

class TRTEngine:
    def __init__(self, engine_path="./pretrained/depth_vits.engine", device='cuda'):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.device = device
        self.diag_counter = 0
        
        # 1. 加载引擎
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 2. 获取节点名称
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # 3. 获取输入输出形状和类型
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        
        # 自动识别 Engine 的数据类型 (非常关键！如果是 FP16 engine 则用 float16)
        trt_type = self.engine.get_tensor_dtype(self.input_name)
        self.dtype = torch.float16 if trt_type == trt.float16 else torch.float32
        
        # 4. 预分配显存 (使用 TensorRT 期待的精确形状)
        self.d_input = torch.empty(self.input_shape, device=device, dtype=self.dtype)
        self.d_output = torch.empty(self.output_shape, device=device, dtype=self.dtype)

    def estimate_depth(self, frame_in):
        # 如果传入的是 Numpy，在这里自动转 Tensor
        if isinstance(frame_in, np.ndarray):
            frame_tensor = torch.as_tensor(frame_in, device=self.device).permute(2, 0, 1).unsqueeze(0).float()
        else:
            frame_tensor = frame_in

        if frame_tensor.dim() == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
            
        _, _, h, w = frame_tensor.shape
        
        with torch.no_grad():
            img_resized = torch.nn.functional.interpolate(
                frame_tensor, size=(728, 728), mode='bilinear', align_corners=False
            )
            img_resized = ((img_resized / 255.0 - 0.5) / 0.5).to(self.dtype)

            self.context.set_tensor_address(self.input_name, img_resized.data_ptr())
            self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())
            self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            
            # 直接在 GPU 上处理
            depth = self.d_output.float()
            depth_resized = torch.nn.functional.interpolate(
                depth.view(1, 1, *self.output_shape[-2:]), 
                size=(h, w), mode='bilinear', align_corners=False
            ) # 保持 [1, 1, H, W] 方便后续拼接

            d_min, d_max = depth_resized.min(), depth_resized.max()
            depth_gpu = (depth_resized - d_min) / (d_max - d_min + 1e-5) * 255.0

            return depth_gpu # 返回 [1, 1, H, W]

    def create_sbs_generic_gpu(self, frame_gpu, depth_gpu, strength=0.6, convergence=0.5, is_half=True):
        # 统一维度
        if frame_gpu.dim() == 3:
            frame_gpu = frame_gpu.unsqueeze(0)
        if depth_gpu.dim() == 2:
            depth_gpu = depth_gpu.unsqueeze(0).unsqueeze(0)
            
        _, _, h, w = frame_gpu.shape
        max_shift = w * 0.02 * strength
        neutral_point = convergence * 255.0
        
        disparity = (depth_gpu - neutral_point) / 255.0 * (max_shift / (w / 2))
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=frame_gpu.device),
            torch.linspace(-1, 1, w, device=frame_gpu.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # 左右眼位移
        shift_l = base_grid.clone()
        shift_l[..., 0] -= disparity.permute(0, 2, 3, 1)[..., 0] * 0.5
        
        shift_r = base_grid.clone()
        shift_r[..., 0] += disparity.permute(0, 2, 3, 1)[..., 0] * 0.5

        left_eye = torch.nn.functional.grid_sample(frame_gpu, shift_l, mode='bilinear', padding_mode='border', align_corners=True)
        right_eye = torch.nn.functional.grid_sample(frame_gpu, shift_r, mode='bilinear', padding_mode='border', align_corners=True)

        # 横向拼接 [1, 3, H, 2W]
        combined = torch.cat([left_eye, right_eye], dim=3)

        # 根据模式输出
        if is_half:
            # 压缩回原始宽度
            return torch.nn.functional.interpolate(combined, size=(h, w), mode='bilinear', align_corners=False)
        else:
            # 保持全宽度
            return combined
