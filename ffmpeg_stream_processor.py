import ffmpeg
import subprocess
import threading
import queue
import cv2
import numpy as np
import time
import sys
import selectors
import torch

class FFmpegStreamProcessor:
    """FFmpeg流处理器 - 音视频自动同步版"""
    
    def __init__(self, input_url, output_url, width=1920, height=1080, 
                 framerate=30, bitrate="4M", output_mode='half-sbs',
                 strength=1.2, convergence=0.5, use_trt=False):

        self.input_url = input_url
        self.output_url = output_url
        self.width = width
        self.height = height
        self.framerate = framerate
        self.bitrate = bitrate
        self.output_mode = output_mode
        self.strength=strength
        self.convergence=convergence
        self.use_trt = use_trt
        
        self.input_process = None
        self.output_process = None
        self.input_thread = None
        self.output_thread = None
        self.process_thread = None
        
        self.frame_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.running = False
        
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'dropped_frames': 0,
            'avg_process_time': 0.0
        }

    def start_input_stream(self):
        """启动输入流读取"""
        args = (
            ffmpeg
            .input(self.input_url, r=self.framerate)
            .filter('scale', self.width, self.height) 
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', vcodec='rawvideo')
            .global_args('-loglevel', 'error')
            .compile()
        )
        print(f"启动输入流: {self.input_url}")
        try:
            self.input_process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
            )
            self.input_thread = threading.Thread(target=self.read_frames_thread, args=(self.input_process.stdout,), daemon=True)
            self.input_thread.start()
            return True
        except Exception as e:
            print(f"输入流启动失败: {e}")
            return False

    def read_frames_thread(self, stdout):
        frame_size = self.width * self.height * 3

        # 1. 初始化选择器，监控 stdout 的读取事件
        selector = selectors.DefaultSelector()
        selector.register(stdout, selectors.EVENT_READ)

        empty_retry_count = 0
        max_empty_retry = 10 # 允许约 10 秒的空数据

        while self.running:
            try:
                # 这会让线程在这一行等待，但不会死锁，1秒没数据就会往下走
                events = selector.select(timeout=1.0)
                
                if not events:
                    # 3. 如果 1 秒内没有任何数据到达
                    empty_retry_count += 1
                    if empty_retry_count >= max_empty_retry:
                        print(f"🛑 [断开] 2D 流已消失超过 {max_empty_retry} 秒，判定为结束")
                        self.running = False
                        break
                    continue # 继续下一轮 select 等待

                # 如果读到空字节（EOF），说明进程已经彻底关了
                in_bytes = stdout.read(frame_size)
                if not in_bytes or len(in_bytes) == 0:
                        self.running = False
                        break

                # 读到数据了，重置计数器
                empty_retry_count = 0

                in_frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                try:
                    self.frame_queue.put(in_frame, block=False)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(in_frame)
                        self.stats['dropped_frames'] += 1
                    except queue.Empty: pass
                self.stats['total_frames'] += 1
            except Exception as e:
                print(f"读取线程异常: {e}")
                self.running = False
                break

        # 退出前注销选择器
        selector.unregister(stdout)
        selector.close()

    def start_output_stream(self):
        """核心：双输入 FFmpeg 启动逻辑 (视频来自管道, 音频来自原流)"""
        # 注意：这里我们手动拼接 ffmpeg 命令，因为 python-ffmpeg 在处理多输入 map 时容易产生路径歧义
        cmd = [
            'ffmpeg', '-y',
            # 输入 0: 来自 Python 管道的处理后视频
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', 
            '-s', f'{self.width}x{self.height}', '-r', str(self.framerate), 
            '-i', '-', 
            # 输入 1: 原始 RTMP 流 (音频源)
            '-fflags', 'nobuffer+genpts', '-flags', 'low_delay', 
            '-i', self.input_url,
            
            # 映射关系
            '-map', '0:v:0',      # 取处理后的视频
            '-map', '1:a:0?',     # 取原始流的音频 (如果有的话)
            
            # --- 视频编码设置 ---
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', 
            '-tune', 'zerolatency', '-b:v', self.bitrate,
            '-g', str(self.framerate * 2), # 设置 GOP 长度，提高 FLV 兼容性
            
            # --- 关键音频修复方案 ---
            '-c:a', 'aac',           # 1. 放弃 copy，强制转码为 AAC (FLV 标准格式)
            '-b:a', '128k',          # 2. 设定固定音频码率
            '-ar', '44100',          # 3. 设定标准采样率 (44.1kHz 在 FLV 中最稳)
            '-ac', '2',              # 4. 强制双声道
            '-af', 'aresample=async=1000:min_hard_comp=0.100000:first_pts=0', # 5. 核心：重采样并强制对齐视频第一帧
            
            # --- 全局同步参数 ---
            '-vsync', 'cfr',         # 强制恒定帧率同步
            '-max_muxing_queue_size', '1024', # 增大混流队列，防止同步时报错
            '-f', 'flv', 
            self.output_url
        ]
        
        print(f"启动同步输出流: {self.output_url}")
        try:
            # 丢弃日志防止管道污染
            self.output_process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
            )
            self.output_thread = threading.Thread(target=self.write_frames_thread, args=(self.output_process.stdin,), daemon=True)
            self.output_thread.start()
            return True
        except Exception as e:
            print(f"输出流启动失败: {e}")
            return False

    def write_frames_thread(self, stdin):
        while self.running:
            try:
                # 1. 尝试从队列拿处理好的 3D 帧
                frame = self.output_queue.get(block=True, timeout=0.1)
                
                # 2. 尺寸检查
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # 3. 核心：尝试写入管道
                try:
                    stdin.write(frame.tobytes())
                    # 对于 RTMP 这种流，如果不是极高码率，建议保留 flush 确保数据发出
                    # stdin.flush() 
                except (BrokenPipeError, OSError) as e:
                    print(f"🛑 [输出] FFmpeg 管道已关闭 (Broken Pipe)，停止推流")
                    self.running = False
                    break

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ 写入线程未知异常: {e}")
                self.running = False
                break
        print("✅ 输出线程已安全退出")


    def process_frames_thread(self, process_func, device):
        while self.running:
            try:
                # 1. 监控：获取帧时的队列状态
                q_size = self.frame_queue.qsize()
                
                raw_frame = self.frame_queue.get(block=True, timeout=1.0)
                
                start_p = time.time()
                processed_frame = process_func(raw_frame, device=device, output_mode=self.output_mode,
                    strength=self.strength, convergence=self.convergence, use_trt=self.use_trt)

                p_time = time.time() - start_p
                
                if processed_frame is not None:
                    self.stats['processed_frames'] += 1
                    self.stats['avg_process_time'] = self.stats['avg_process_time'] * 0.9 + p_time * 0.1
                    try:
                        self.output_queue.put(processed_frame, block=False)
                    except queue.Full:
                        try:
                            self.output_queue.get_nowait()
                            self.output_queue.put_nowait(processed_frame)
                        except: pass
            except queue.Empty: continue

    def start(self, process_func, device='cpu'):
        self.running = True
        if not self.start_input_stream(): return False
        if not self.start_output_stream(): return False
        self.process_thread = threading.Thread(target=self.process_frames_thread, 
            args=(process_func, device), daemon=True)
        self.process_thread.start()
        return True

    def stop(self):
        """彻底清理资源，确保下一次重连成功"""
        self.running = False
        print("🧹 正在清理流处理器资源...")

        # 1. 强制杀掉所有 FFmpeg 进程
        for p in [self.input_process, self.output_process]:
            if p:
                try:
                    p.stdin.close() if p.stdin else None
                    p.terminate()
                    p.wait(timeout=2) # 等待进程完全释放端口
                except:
                    if p: p.kill() # 强制杀死

        # 2. 关键：清空队列，防止旧数据污染新流
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: break
        while not self.output_queue.empty():
            try: self.output_queue.get_nowait()
            except: break

        # 3. 释放显存
        torch.cuda.empty_cache()
        
        print("✨ 资源清理完成，准备进入下一轮侦听")

    def get_status(self):
        return {
            'stats': self.stats.copy(),
            'queue_sizes': {'input': self.frame_queue.qsize(), 'output': self.output_queue.qsize()}
        }
