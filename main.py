#!/usr/bin/env python3

import argparse
import time
import signal
from video_depth_anything import process_frame, load_model
from ffmpeg_stream_processor import FFmpegStreamProcessor
import torch
import sys
import subprocess
import re

def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

import subprocess
import re
import time

def get_stream_info(url):
    """
    稳健版流探测：
    1. 使用 Popen 配合 communicate 避免管道阻塞
    2. 增加 3 秒硬性超时，防止脚本卡死
    """
    cmd = [
        'ffprobe', 
        '-v', 'error',
        '-probesize', '100000',
        '-analyzeduration', '0',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', 
        '-of', 'csv=p=0',
        url
    ]
    try:
        # 启动进程
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 核心：设置超时时间为 3 秒
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill() # 超时强制杀掉进程
            return False, None, None

        if proc.returncode != 0:
            return False, None, None

        output = stdout.decode().strip()
        # 匹配终端输出的 "720,1280" 格式
        match = re.search(r'(\d+),(\d+)', output)
        if match:
            w, h = int(match.group(1)), int(match.group(2))
            return True, w, h
            
    except Exception as e:
        # 静默失败，返回 False 让主循环继续等待
        pass
        
    return False, None, None

def main():
    parser = argparse.ArgumentParser(
        description="Video Depth Anything RTMP 2D转3D流处理器",
        add_help=False
    )

    parser.add_argument("--help", action="help") 
    # 输入参数
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="输入RTMP流地址 (例如: rtmp://localhost:1935/live/input_stream)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="输出RTMP流地址 (例如: rtmp://localhost:1935/live/output_stream)"
    )
    
    parser.add_argument(
        "-r", "--framerate", 
        type=int, 
        default=30, 
        help="输出视频帧率 (默认: 30)"
    )
    
    parser.add_argument(
        "-b", "--bitrate", 
        default="3M", 
        help="输出视频比特率 (默认: 3M)"
    )
    
    # 处理参数
    parser.add_argument(
        "-m", "--mode", 
        default="half-sbs", 
        choices=["half-sbs", "sbs"],
        help="输出模式 (half-sbs: 左右格式压缩, sbs: 左右格式)"
    )
    
    parser.add_argument(
        "-d", "--device", 
        default="auto", 
        help="处理设备 (auto/cpu/cuda/mps)"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="显示详细调试信息"
    )

    parser.add_argument('-s', '--strength', type=float, default=1.2, help='3D深度强度 (建议 0.5-2.0)')
    parser.add_argument('-c', '--convergence', type=float, default=0.5, help='会聚平面 (0.0全出屏, 1.0全入屏)')
    parser.add_argument('--trt', action='store_true', help='开启动 TensorRT + CuPy 加速')

    args = parser.parse_args()

    print("=" * 60)
    print("Video Depth Anything 2D转3D流处理器")
    print("=" * 60)
    print(f"输入流: {args.input}")
    print(f"输出流: {args.output}")
    print(f"视频尺寸: {args.framerate}fps")
    print(f"比特率: {args.bitrate}")
    print(f"输出模式: {args.mode}")
    print(f"3D深度强度: {args.strength}")
    print(f"会聚平面: {args.convergence}")
    print(f"使用 TensorRT 加速: {'是' if args.trt else '否'}")
    
    # 设备配置
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
        
    print(f"处理设备: {device}")

    # 加载模型
    load_model(device=device, use_trt=args.trt)

    processor = None 
    # 信号处理
    def signal_handler(sig, frame):
        print("\n👋 收到停止信号，系统正在退出...")
        if processor is not None:
            processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🛰️  3D 转换服务已就绪，进入循环侦听模式...")

    # 启动流处理
    while True:
        # 第一步：探测 2D 源是否存在和分辨率
        is_alive, stream_w, stream_h = get_stream_info(args.input)
        
        if not is_alive:
            # 使用 \r 实时更新时间，不刷屏
            print(f"⏳ [{time.strftime('%H:%M:%S')}] 等待 2D 信号源: {args.input}", end='\r')
            time.sleep(2) # 4090 性能强，可以缩短轮询间隔
            continue
    
        print(f"\n✅ 检测到 2D 信号! 分辨率: {stream_w}x{stream_h}")
        # 第二步：创建新的流处理器实例
        processor = FFmpegStreamProcessor(
            input_url=args.input,
            output_url=args.output,
            width=stream_w,
            height=stream_h,
            framerate=args.framerate,
            bitrate=args.bitrate,
            output_mode=args.mode,
            strength=args.strength,
            convergence=args.convergence,
            use_trt=args.trt
        )

        try:
            # 第三步：启动流处理
            if not processor.start(process_frame, device=device):
                print("❌ 启动处理器失败，5秒后重试")
                time.sleep(5)
                continue
            
            print(f"🎬 3D 推流成功: {args.output}")
            
            # 第四步：实时监控循环
            last_print = time.time()
            while processor.running:
                # 检查 FFmpeg 进程是否还在运行
                if processor.input_process and processor.input_process.poll() is not None:
                    print("\n⚠️  检测到 2D 输入进程意外结束")
                    break
                
                time.sleep(1)
                
                # 打印性能状态
                if time.time() - last_print >= 5:
                    stats = processor.get_status()
                    print(f"\n📊 状态报告 ({time.strftime('%H:%M:%S')}):")
                    print(f"  帧数统计: 总入队 {stats['stats']['total_frames']} | 已处理 {stats['stats']['processed_frames']} | 丢弃 {stats['stats']['dropped_frames']}")
                    print(f"  性能数据: {stats['stats']['avg_process_time']:.3f}s/帧 | 队列: In({stats['queue_sizes']['input']}) Out({stats['queue_sizes']['output']})")
                    last_print = time.time()
                    
        except Exception as e:
            print(f"\n💥 运行异常: {e}")
        finally:
            # 第五步：清理当前连接，等待下一次循环
            print("🧹 正在清理当前推流连接，准备进入下一轮侦听...")
            if processor:
                processor.stop()
                processor = None
            time.sleep(2) # 留给 RTMP 服务器一点断开时间

if __name__ == "__main__":
    main()
