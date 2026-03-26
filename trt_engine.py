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

    def estimate_depth(self, frame):
        h, w = frame.shape[:2]
        with torch.no_grad():
            # 1. 预处理：BGR -> RGB -> Tensor -> 归一化 -> Resize
            # 注意：在 GPU 上完成所有操作以保证速度
            img_gpu = torch.as_tensor(frame, device=self.device).permute(2, 0, 1).contiguous().float()
            
            # 缩放至模型输入尺寸 (518, 518)
            img_resized = torch.nn.functional.interpolate(
                img_gpu.unsqueeze(0), size=(518, 518), mode='bilinear', align_corners=False
            )
            
            # 归一化 (匹配你之前的测试脚本: /0.5 - 1.0 等价于 /255 然后 (x-0.5)/0.5)
            img_resized = (img_resized / 255.0 - 0.5) / 0.5
            img_resized = img_resized.to(self.dtype)

            # 2. 确保地址绑定 (部分 TRT 版本在 Context 切换时会丢失绑定，建议显式设置)
            self.context.set_tensor_address(self.input_name, img_resized.data_ptr())
            self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())

            # 3. 执行推理
            # 使用当前显存流，避免不必要的同步
            self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            
            # 4. 后处理
            depth = self.d_output.clone() # 拷贝一份防止被下一帧覆盖
            
            # 移除 batch 和 channel 维度
            if depth.dim() == 4:
                depth = depth.squeeze(0).squeeze(0)
            elif depth.dim() == 3:
                depth = depth.squeeze(0)

            # 5. 缩放回原图尺寸
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze()

            # 6. 归一化到 0-255
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-5:
                depth = (depth - d_min) / (d_max - d_min) * 255.0
            else:
                depth = torch.zeros_like(depth)

            # 调试信息
            if self.diag_counter % 50 == 0:
                debug_img = depth.cpu().numpy().astype(np.uint8)
                cv2.imwrite("./pretrained/debug_depth.png", debug_img)
            self.diag_counter += 1

            return depth.cpu().numpy().astype(np.uint8)

    # def create_half_sbs_gpu(self, frame, depth_gpu, strength=0.6, convergence=0.5):
    #     h, w = frame.shape[:2]
        
    #     # 1. 强制使用 float32 提高采样精度
    #     frame_cp = cp.asarray(frame).astype(cp.float32)
    #     depth_cp = cp.asarray(depth_gpu).astype(cp.float32)

    #     # 2. 视差计算 (严格对齐原始公式)
    #     max_shift = w * 0.02 * strength
    #     neutral_point = convergence * 255.0
    #     disparity = (depth_cp - neutral_point) / 255.0 * max_shift
        
    #     y, x = cp.mgrid[0:h, 0:w].astype(cp.float32)

    #     # 3. 左右位移映射
    #     map_l_x = cp.clip(x - disparity * 0.5, 0, w - 1)
    #     map_r_x = cp.clip(x + disparity * 0.5, 0, w - 1)

    #     # 4. 执行高精度采样
    #     coords_l = cp.stack([y, map_l_x])
    #     coords_r = cp.stack([y, map_r_x])
        
    #     left_eye = cp.empty_like(frame_cp)
    #     right_eye = cp.empty_like(frame_cp)
        
    #     for i in range(3):
    #         # 强制 order=1 (线性) 并检查数据连续性
    #         left_eye[..., i] = map_coordinates(frame_cp[..., i], coords_l, order=1, mode='nearest')
    #         right_eye[..., i] = map_coordinates(frame_cp[..., i], coords_r, order=1, mode='nearest')

    #     # 5. 拼接
    #     combined = cp.hstack([left_eye, right_eye])
        
    #     # --- 画质关键：回到 CPU 后使用高画质缩放 ---
    #     combined_cpu = cp.asnumpy(combined).astype(np.uint8)
        
    #     # 换用 INTER_CUBIC (双三次插值)，它比 INTER_LINEAR 锐利得多
    #     return cv2.resize(combined_cpu, (w, h), interpolation=cv2.INTER_CUBIC)


    # TODO: 暂未修改效果不佳
    def create_sbs_gpu(self, frame, depth_gpu, strength=0.6, convergence=0.5):
        """全 GPU 高精度视差拼接 (2W x H)"""
        h, w = frame.shape[:2]
        
        # 1. 确保输入数据为 float32 精度，防止 fp16 导致的计算舍入模糊
        # frame_torch: [1, 3, H, W]
        frame_torch = torch.as_tensor(frame.copy(), device='cuda').permute(2, 0, 1).unsqueeze(0).float()
        depth_f32 = depth_gpu.float() # 深度图转为 f32 确保位移计算精准

        # 2. 计算像素级位移 (Disparity)
        max_shift = w * 0.02 * strength
        neutral_point = convergence * 255.0
        disparity = (depth_f32 - neutral_point) / 255.0 * max_shift
        
        # 3. 构建标准化的采样网格 (Grid)
        # PyTorch 的 grid 范围必须在 [-1.0, 1.0]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device='cuda'),
            torch.linspace(-1, 1, w, device='cuda'),
            indexing='ij'
        )

        # 核心：将像素位移转换为归一化单位 (2.0 / (w-1) 表示单像素在 [-1,1] 间的跨度)
        shift_norm = (disparity * 0.5) * (2.0 / (w - 1))

        # 4. 生成左右眼网格：[1, H, W, 2]
        # grid_l 采样坐标向左偏移，grid_r 采样坐标向右偏移
        grid_l = torch.stack([grid_x - shift_norm, grid_y], dim=-1).unsqueeze(0)
        grid_r = torch.stack([grid_x + shift_norm, grid_y], dim=-1).unsqueeze(0)

        # 5. 高质量采样 (使用 bicubic 替代 bilinear 解决模糊问题)
        # 修正：padding_mode 使用 'border' 代替 'replicate'
        left_eye = torch.nn.functional.grid_sample(
            frame_torch, grid_l, mode='bicubic', padding_mode='border', align_corners=True
        )
        right_eye = torch.nn.functional.grid_sample(
            frame_torch, grid_r, mode='bicubic', padding_mode='border', align_corners=True
        )

        # 6. 横向拼接并返回 [1, 3, H, 2W]
        return torch.cat([left_eye, right_eye], dim=-1)
