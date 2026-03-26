import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
import time
from trt_engine import TRTEngine

sys.path.append(os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))

from depth_anything_v2.dpt import DepthAnythingV2

# 全局变量
model = None
engine_instance = None

def load_model(device='cuda', use_trt=False):
    """加载 Video Depth Anything V2 Small (ViT-S)"""
    global engine_instance, model

    if use_trt:
        print(f"🚀 正在加载 TensorRT 引擎")
        engine_instance = TRTEngine(device=device)
        return engine_instance
    else:
        print(f"正在加载 Video Depth Anything V2 Small (设备: {device})...")
        # 1. 关键：ViT-Small 的官方配置参数
        model_configs = {
            'encoder': 'vits', 
            'features': 64, 
            'out_channels': [48, 96, 192, 384]
        }
        
        # 2. 初始化架构
        model = DepthAnythingV2(**model_configs)
        
        # 3. 权重路径处理
        path = './pretrained/depth_anything_v2_vits.pth'
        if not os.path.exists(path):
            path = 'depth_anything_v2_vits.pth'
        
        if os.path.exists(path):
            try:
                # 解决 PyTorch 2.6+ 兼容性问题
                state_dict = torch.load(path, map_location='cpu', weights_only=False)
                model.load_state_dict(state_dict)
                print(f"成功加载 ViT-S 视频权重: {path}")
            except Exception as e:
                print(f"权重加载失败: {e}")
                sys.exit(1)
        else:
            print(f"错误: 找不到权重文件 {path}，请从 HuggingFace 下载")
            sys.exit(1)

        model.to(device).eval()
        return model

def estimate_depth(frame, device='cuda'):
    """高性能 Forward 版本：全 GPU 预处理与后处理"""
    global model
    h, w = frame.shape[:2]
    
    t0 = time.time()

    # 1. 快速预处理 (利用 Tensor 加速)
    # 将 BGR 转换为 RGB 并转为 Float Tensor 直接送入 GPU
    with torch.no_grad():
        # [H, W, C] -> [C, H, W] -> [1, C, H, W]
        img = torch.from_numpy(frame.copy()).to(device).float().permute(2, 0, 1).unsqueeze(0)

        img = img / 255.0  # 归一化
        
        # 强制缩放到模型要求的推理尺寸 (如 518)
        # 这一步在 GPU 上做比 cv2.resize 快得多
        inference_size = 518
        img = torch.nn.functional.interpolate(img, size=(inference_size, inference_size), mode='bilinear', align_corners=False)
        
        t_pre = time.time()

        # 2. 核心推理 (FP16 加速)
        with torch.amp.autocast('cuda'):
            # 直接调用 model(img) 即执行 forward
            # 输出通常是 [1, H_small, W_small]
            depth = model(img)
            
        # 必须同步才能测准真实物理耗时
        torch.cuda.synchronize()
        t_infer = time.time()

        # 3. 后处理：在 GPU 上直接缩放回原图尺寸
        # 避免了在 CPU 上做大图 resize
        depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze()
        
        # 4. 快速归一化 (也在 GPU 上完成)
        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min > 1e-5:
            depth = (depth - d_min) / (d_max - d_min) * 255.0
        else:
            depth = torch.zeros_like(depth)

        # 5. 唯一下传：只把结果下传回 CPU
        depth_np = depth.cpu().numpy().astype(np.uint8)
        
    t_end = time.time()

    # 监控输出 (每100帧)
    if getattr(estimate_depth, 'counter', 0) % 100 == 0:
        print(f"  [Forward监控] 总计: {t_end-t0:.4f}s")
        print(f"    ├─ GPU预处理: {t_pre-t0:.4f}s")
        print(f"    ├─ 纯推理(Forward): {t_infer-t_pre:.4f}s")
        print(f"    └─ GPU后处理+归一化: {t_end-t_infer:.4f}s")
    estimate_depth.counter = getattr(estimate_depth, 'counter', 0) + 1
        
    return depth_np

def create_sbs_generic(frame, depth, strength=0.6, convergence=0.5, is_half=True):
    """
    通用 SBS 生成函数
    :param is_half: True 为 Half-SBS (1920x1080), False 为 Full-SBS (3840x1080)
    """
    h, w = frame.shape[:2]
    # 1. 计算最大位移量 (以像素为单位)
    max_shift = w * 0.02 * strength
    
    # 2. 计算视差图 (Disparity Map)
    neutral_point = convergence * 255.0
    disparity = (depth.astype(np.float32) - neutral_point) / 255.0 * max_shift
    
    # 3. 生成基础网格坐标
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # 4. 左右眼重映射坐标计算 (矢量化操作，极快)
    map_l_x = np.clip(x - disparity * 0.5, 0, w - 1)
    map_r_x = np.clip(x + disparity * 0.5, 0, w - 1)

    # 5. 执行重映射 (Remap)
    left_eye = cv2.remap(frame, map_l_x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    right_eye = cv2.remap(frame, map_r_x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 6. 横向拼接得到 Full-SBS (2W x H)
    combined = np.hstack([left_eye, right_eye])

    # 7. 根据模式返回结果
    if is_half:
        # 压缩宽度到 1xW (左右眼各占一半宽度)
        return cv2.resize(combined, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # 保持 2xW (全尺寸左右格式)
        return combined


def process_frame(frame, device='cuda', output_mode='half-sbs', strength=1.2, convergence=0.5, use_trt=False, **kwargs):
    """主处理函数 - 精简监控版"""
    try:
        t_start = time.time()
        t_infer = t_start
        
        if use_trt and engine_instance:
            # 1. 先转换：Numpy (H,W,C) BGR -> Tensor (1,3,H,W) RGB
            frame_gpu = torch.from_numpy(np.ascontiguousarray(frame)).to(device).permute(2, 0, 1).unsqueeze(0).float()
            
            # 2. 传入转换后的 Tensor
            depth_gpu = engine_instance.estimate_depth(frame_gpu) 
            
            t_infer = time.time()
            
            if output_mode == 'half-sbs':
                out_gpu = engine_instance.create_sbs_generic_gpu(frame_gpu, depth_gpu, strength, convergence, is_half=True)
                # 统一转回 Numpy
                out = out_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            elif output_mode == 'sbs':
                # 同样建议实现全宽度的 create_sbs_gpu 以保持高性能
                out_gpu = engine_instance.create_sbs_generic_gpu(frame_gpu, depth_gpu, strength, convergence, is_half=False)
                out = out_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                # 伪彩色映射 (目前在 CPU 上更快)
                out = cv2.applyColorMap(depth_gpu.squeeze().cpu().numpy().astype(np.uint8), cv2.COLORMAP_MAGMA)

        else:
            # 1. GPU 核心流程：包含 GPU 缩放 + 推理 + GPU 尺寸恢复
            # 直接传入原图，由 estimate_depth 内部完成所有显存内操作
            depth = estimate_depth(frame, device=device)
            t_infer = time.time()
            
            # 2. SBS 拼接 (CPU 核心耗时)
            if output_mode == 'half-sbs':
                out = create_sbs_generic(frame, depth, strength, convergence, is_half=True)
            elif output_mode == 'sbs':
                out = create_sbs_generic(frame, depth, strength, convergence, is_half=False)
            else:
                out = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)

        t_sbs = time.time()
        # --- 性能监控打印：每 100 帧输出一次 ---
        if getattr(process_frame, 'counter', 0) % 100 == 0:
            total_time = t_sbs - t_start
            print(f"\n[性能监控] 总耗时: {total_time:.4f}s")
            print(f"  └─ GPU 核心流程: {t_infer - t_start:.4f}s")
            print(f"  └─ SBS 拼接(CPU): {t_sbs - t_infer:.4f}s")
            
            # 如果总耗时超过 0.033s (30FPS 临界值)，打印预警
            if total_time > 0.033:
                print(f"  ⚠️  警告: 当前处理速度低于 30 FPS")
        
        process_frame.counter = getattr(process_frame, 'counter', 0) + 1
        return out

    except Exception as e:
        import traceback
        print(f"❌ 处理出错: {e}")
        traceback.print_exc()
        return None
