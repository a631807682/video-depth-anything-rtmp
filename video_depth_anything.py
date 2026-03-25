import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))

from depth_anything_v2.dpt import DepthAnythingV2

# 全局变量
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(device='cuda'):
    """加载 Video Depth Anything V2 Small (ViT-S)"""
    global model
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
            
        # 4090 必须同步才能测准真实物理耗时
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

# --- 3D 转换逻辑 (SBS 模式) ---
def create_sbs(frame, depth, strength=1.5):
    """
    创建左右格式 (Side-by-Side)
    注意：输出宽度将翻倍 (3840x1080)
    """
    h, w = frame.shape[:2]
    # 生成视差（深度越大，位移越大）
    disparity = (depth.astype(np.float32) / 255.0) * strength * 20.0
    
    # 创建左右眼视图
    left_eye = np.zeros_like(frame)
    right_eye = np.zeros_like(frame)
    
    for i in range(h):
        shift = disparity[i].astype(np.int32)
        # 简易视差偏移映射
        left_eye[i, :] = np.roll(frame[i, :], -5, axis=0) # 基础偏移
        right_eye[i, :] = np.roll(frame[i, :], 5, axis=0)
        
    # 横向拼接
    return np.hstack([left_eye, right_eye])

def create_sbs_half(frame, depth, strength=0.6, convergence=0.5):
    """
    convergence: 0.0 ~ 1.0 (0.0=全出屏, 1.0=全入屏, 0.5=平衡)
    """
    h, w = frame.shape[:2]
    max_shift = w * 0.02 * strength
    
    # 手动定义中性面：0.5 表示深度图中值处在屏幕平面
    neutral_point = convergence * 255.0
    disparity = (depth.astype(np.float32) - neutral_point) / 255.0 * max_shift
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # 3. 左右眼对半分担位移 (各平移 0.5 倍，总位移不变但拉伸感减小)
    map_l_x = np.clip(x - disparity * 0.5, 0, w - 1)
    map_r_x = np.clip(x + disparity * 0.5, 0, w - 1)

    # 4. 重映射
    left_eye = cv2.remap(frame, map_l_x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    right_eye = cv2.remap(frame, map_r_x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 5. SBS 拼接
    combined = np.hstack([left_eye, right_eye])
    return cv2.resize(combined, (w, h), interpolation=cv2.INTER_LINEAR)

def process_frame(frame, device='cuda', output_mode='half-sbs', **kwargs):
    """主处理函数 - 精简监控版"""
    try:
        global frame_count
        frame_count += 1

        t_start = time.time()
        
        # 1. GPU 核心流程：包含 GPU 缩放 + 推理 + GPU 尺寸恢复
        # 直接传入原图，由 estimate_depth 内部完成所有显存内操作
        depth = estimate_depth(frame, device=device)
        t_infer = time.time()
        
        # 2. SBS 拼接 (CPU 核心耗时)
        if output_mode == 'half-sbs':
            out = create_sbs_half(frame, depth, strength=1.2)
        elif output_mode == 'sbs':
            out = create_sbs(frame, depth, strength=0.8)
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
