#!/usr/bin/env python3

import argparse
import time
import signal
from video_depth_anything import process_frame, load_model
from ffmpeg_stream_processor import FFmpegStreamProcessor
import torch
import sys

def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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
    
    # 视频参数
    parser.add_argument(
        "-w", "--width", 
        type=int, 
        default=1280, 
        help="输出视频宽度 (默认: 1280)"
    )
    
    parser.add_argument(
        "-h", "--height", 
        type=int, 
        default=720, 
        help="输出视频高度 (默认: 720)"
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
        default="sbs-half", 
        choices=["sbs-half", "sbs"],
        help="输出模式 (sbs-half: 左右格式压缩, sbs: 左右格式)"
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
    
    args = parser.parse_args()

    print("=" * 60)
    print("Video Depth Anything 2D转3D流处理器")
    print("=" * 60)
    print(f"输入流: {args.input}")
    print(f"输出流: {args.output}")
    print(f"视频尺寸: {args.width}x{args.height} @ {args.framerate}fps")
    print(f"比特率: {args.bitrate}")
    print(f"输出模式: {args.mode}")
    
    # 设备配置
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
        
    print(f"处理设备: {device}")

    # 加载模型
    load_model(device=device)

    # 创建流处理器
    processor = FFmpegStreamProcessor(
        input_url=args.input,
        output_url=args.output,
        width=args.width,
        height=args.height,
        framerate=args.framerate,
        bitrate=args.bitrate,
        output_mode=args.mode
    )

    # 信号处理
    def signal_handler(sig, frame):
        print("\n收到停止信号，正在清理...")
        processor.stop()
        print("系统已停止")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动流处理
    try:
        if not processor.start(process_frame, device=device):
            print("启动流处理失败")
            return
        
        print("\n" + "-" * 60)
        print("流处理正在运行...")
        print("按 Ctrl+C 停止")
        print("-" * 60)
        
        # 主循环 - 定期打印状态
        last_print = time.time()
        while processor.running:
            time.sleep(1)
            
            if time.time() - last_print >= 5:
                stats = processor.get_status()
                print(f"\n状态更新 ({time.strftime('%H:%M:%S')}):")
                print(f"  总帧数: {stats['stats']['total_frames']}")
                print(f"  处理帧数: {stats['stats']['processed_frames']}")
                print(f"  丢弃帧数: {stats['stats']['dropped_frames']}")
                print(f"  平均处理时间: {stats['stats']['avg_process_time']:.3f}秒/帧")
                print(f"  队列: 输入{stats['queue_sizes']['input']}, 输出{stats['queue_sizes']['output']}, {stats['stats']['total_latency']}")

                last_print = time.time()
                
    except Exception as e:
        print(f"\n运行时错误: {e}")
        processor.stop()

    finally:
        processor.stop()

if __name__ == "__main__":
    main()
