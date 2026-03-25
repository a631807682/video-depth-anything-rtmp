import tensorrt as trt
import torch
import cupy as cp
import cv2
import numpy as np
from cupyx.scipy.ndimage import map_coordinates

class TRTEngine:
    def __init__(self, engine_path="./pretrained/depth_vits_fp16.engine", device='cuda'):
        """初始化 TensorRT 10 引擎并预分配显存"""
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.device = device
        
        # 1. 加载并反序列化 Engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # 2. 创建推理上下文
        self.context = self.engine.create_execution_context()

        # 3. 自动探测 TensorRT 10 的 I/O 节点名称
        self.input_name = None
        self.output_name = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        print(f"✅ TRT 10 节点锁定 -> 输入: '{self.input_name}', 输出: '{self.output_name}'")

        # 4. 获取形状并预分配显存 (使用 PyTorch 管理)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # 预分配连续显存空间，确保推理流水线无阻塞
        self.d_input = torch.empty(self.input_shape, device=device, dtype=torch.float16)
        self.d_output = torch.empty(self.output_shape, device=device, dtype=torch.float16)

        # 5. TensorRT 10 地址绑定 (替代旧版的 bindings 列表)
        self.context.set_tensor_address(self.input_name, self.d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())

    def estimate_depth(self, frame):
        """全 GPU 预处理与深度估计"""
        h, w = frame.shape[:2]
        with torch.no_grad():
            # 1. 彻底解决 UserWarning：先 copy 再转 tensor
            # frame.copy() 确保内存可写，避免 PyTorch 报错
            img_gpu = torch.as_tensor(frame.copy(), device=self.device).permute(2, 0, 1).unsqueeze(0).contiguous().half() / 255.0
            
            # 2. GPU 高速缩放
            img_resized = torch.nn.functional.interpolate(img_gpu, size=(518, 518), mode='bilinear', align_corners=False)
            
            # 3. 拷贝到预绑定的输入缓冲区
            self.d_input.copy_(img_resized)

            # 4. --- 关键修复：使用 TRT 10 异步推理接口 ---
            # 在 TensorRT 10 中，如果你使用了 set_tensor_address，
            # 必须使用 execute_async_v3。0 代表使用默认 CUDA 流。
            self.context.execute_async_v3(0) 
            
            # 5. 同步并后处理
            torch.cuda.synchronize()
            
            # 直接在 GPU 上缩回原图尺寸
            depth = torch.nn.functional.interpolate(self.d_output.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze()
            
            # 快速归一化
            d_min, d_max = depth.min(), depth.max()
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-5) * 255.0
            
            return depth_norm


    def create_sbs_gpu(self, frame, depth_gpu, strength=0.6, convergence=0.5):
        """全 GPU 视差拼接 (2W x H)：利用 CuPy 并行处理海量像素"""
        h, w = frame.shape[:2]
        
        # 将原图异步上传到显存 (仅在必要时执行)
        frame_gpu = cp.asarray(frame) 
        # 将深度图从 Torch 零拷贝转换为 CuPy 数组
        depth_cp = cp.asarray(depth_gpu) 
        
        # 核心算法：计算视差偏移
        max_shift = w * 0.02 * strength
        neutral_point = convergence * 255.0
        disparity = (depth_cp - neutral_point) / 255.0 * max_shift
        
        # GPU 上生成像素坐标网格
        y, x = cp.mgrid[0:h, 0:w].astype(cp.float32)
        
        # 根据深度信息计算左右眼映射坐标
        map_l_x = cp.clip(x - disparity * 0.5, 0, w - 1)
        map_r_x = cp.clip(x + disparity * 0.5, 0, w - 1)
        
        # 使用 CuPy 并行插值采样 (模拟 cv2.remap 的 GPU 行为)
        coords_l = cp.stack([y, map_l_x])
        coords_r = cp.stack([y, map_r_x])
        
        left_eye = cp.empty_like(frame_gpu)
        right_eye = cp.empty_like(frame_gpu)
        for i in range(3): # BGR 三通道独立并行采样
            left_eye[..., i] = map_coordinates(frame_gpu[..., i], coords_l, order=1)
            right_eye[..., i] = map_coordinates(frame_gpu[..., i], coords_r, order=1)
        
        # 显存内水平拼接
        return cp.hstack([left_eye, right_eye])

    def create_half_sbs_gpu(self, frame, depth_gpu, strength=0.6, convergence=0.5):
        """半宽 SBS 生成：全 GPU 完成拼接和压缩"""
        # 1. 获得 GPU 上的双倍宽画面
        full_sbs_gpu = self.create_sbs_gpu(frame, depth_gpu, strength, convergence)
        
        # 2. 转换回 Torch 进行 GPU 高速缩放 (避免 CPU resize 的巨大延迟)
        full_sbs_torch = torch.as_tensor(full_sbs_gpu, device='cuda').permute(2, 0, 1).unsqueeze(0).float()
        
        # 3. 在 GPU 上水平压缩回原始宽度
        h, w = frame.shape[:2]
        half_sbs_torch = torch.nn.functional.interpolate(full_sbs_torch, size=(h, w), mode='bilinear', align_corners=False)
        
        # 4. 仅在最终环节下传回 CPU，转换为 uint8 以便 OpenCV 推流/显示
        return half_sbs_torch.squeeze().permute(1, 2, 0).byte().cpu().numpy()
