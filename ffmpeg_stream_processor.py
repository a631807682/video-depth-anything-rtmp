import ffmpeg
import subprocess
import threading
import queue
import cv2
import numpy as np
import time
import sys

class FFmpegStreamProcessor:
    """FFmpeg流处理器 - 用于RTMP输入输出流处理"""
    
    def __init__(self, input_url, output_url, 
                 width=1920, height=1080, 
                 framerate=30, bitrate="4M",
                 output_mode='sbs-half'):
        self.input_url = input_url
        self.output_url = output_url
        self.width = width
        self.height = height
        self.framerate = framerate
        self.bitrate = bitrate
        self.output_mode = output_mode
        
        self.input_process = None
        self.output_process = None
        
        # --- 核心修改 1: 队列长度设为 1，彻底消除积压延迟 ---
        self.frame_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
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
            # 强制缩放：防止分辨率不对乱码
            .filter('scale', self.width, self.height) 
            .output('pipe:', format='rawvideo', pix_fmt='bgr24',
                   vcodec='rawvideo')
            .global_args('-loglevel', 'error')
            .compile()
        )
        
        print(f"启动输入流: {self.input_url}")
        
        try:
            self.input_process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            self.input_thread = threading.Thread(
                target=self.read_frames_thread,
                args=(self.input_process.stdout,),
                daemon=True
            )
            self.input_thread.start()
            
            return True
        except Exception as e:
            print(f"输入流启动失败: {e}")
            return False

    def read_frames_thread(self, stdout):
        """从FFmpeg读取原始帧的线程 - 改为非阻塞模式"""
        frame_size = self.width * self.height * 3
        while self.running:
            try:
                in_bytes = stdout.read(frame_size)
                if not in_bytes: break
                
                in_frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                
                # --- 核心修改 2: 抢占式入队 ---
                # 如果 GPU 还没处理完，就扔掉旧帧，塞入最新帧
                try:
                    self.frame_queue.put_nowait(in_frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait() # 踢出旧帧
                        self.frame_queue.put_nowait(in_frame) # 放入新帧
                        self.stats['dropped_frames'] += 1
                    except queue.Empty: pass
                
                self.stats['total_frames'] += 1
            except Exception as e:
                print(f"读取帧失败: {e}")
                break

    def start_output_stream(self):
        """启动输出流写入"""
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24',
                s=f'{self.width}x{self.height}', r=self.framerate)
            .output(self.output_url,
                vcodec='libx264', 
                r=self.framerate, # 输出强制 30fps
                pix_fmt='yuv420p',
                preset='ultrafast', # 4090 配合 ultrafast 延迟最低
                tune='zerolatency', # 开启零延迟模式
                x264opts='keyint=30:min-keyint=30', # 每秒一个关键帧，利于秒开
                f='flv')
            .global_args('-loglevel', 'error', '-vsync', 'cfr') # 强制恒定帧率
            .compile()
        )
            
        print(f"启动输出流: {self.output_url}")
        
        try:
            self.output_process = subprocess.Popen(
                args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            self.output_thread = threading.Thread(
                target=self.write_frames_thread,
                args=(self.output_process.stdin,),
                daemon=True
            )
            self.output_thread.start()
            
            return True
        except Exception as e:
            print(f"输出流启动失败: {e}")
            return False

    def write_frames_thread(self, stdin):
        """向FFmpeg写入帧的线程"""
        while self.running:
            try:
                # 保持较短 timeout 增加响应性
                frame = self.output_queue.get(block=True, timeout=0.05)
                
                # --- 核心修改 3: 移除重复的 resize ---
                # 如果 process_frame 已经产出了 1920x1080，这里直接写，节省 CPU
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                stdin.write(frame.tobytes())
                stdin.flush()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"写入帧失败: {e}")
                break

    def process_frames_thread(self, process_func, device='cuda'):
        """处理帧的线程"""
        while self.running:
            try:
                raw_frame = self.frame_queue.get(block=True, timeout=1.0)
                
                start_process = time.time()
                # 调用你带监控的 process_frame
                processed_frame = process_func(raw_frame, device=device, output_mode=self.output_mode)
                process_time = time.time() - start_process
                
                if processed_frame is None: continue

                self.stats['processed_frames'] += 1
                # 滑动平均统计
                self.stats['avg_process_time'] = (self.stats['avg_process_time'] * 0.9 + process_time * 0.1)
                
                # --- 核心修改 4: 非阻塞放入输出队列 ---
                try:
                    self.output_queue.put_nowait(processed_frame)
                except queue.Full:
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(processed_frame)
                    except queue.Empty: pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理帧失败: {e}")
                continue

    def start(self, process_func, device='cpu'):
        """启动完整流程"""
        self.running = True
        
        # 启动输入流
        if not self.start_input_stream():
            self.running = False
            return False
        
        # 启动输出流
        if not self.start_output_stream():
            self.running = False
            if self.input_process:
                self.input_process.terminate()
            return False
        
        # 启动处理线程
        self.process_thread = threading.Thread(
            target=self.process_frames_thread,
            args=(process_func, device),
            daemon=True
        )
        self.process_thread.start()
        
        print("流处理系统启动成功")
        return True

    def stop(self):
        """停止所有流处理"""
        self.running = False
        
        if self.input_process:
            self.input_process.terminate()
            self.input_process = None
            
        if self.output_process:
            self.output_process.terminate()
            self.output_process = None
            
        if self.input_thread:
            self.input_thread.join(timeout=2)
            self.input_thread = None
            
        if self.output_thread:
            self.output_thread.join(timeout=2)
            self.output_thread = None
            
        if hasattr(self, 'process_thread') and self.process_thread:
            self.process_thread.join(timeout=2)
            self.process_thread = None
            
        print("流处理系统已停止")

    def get_status(self):
        """获取状态信息"""
        status = {
            'running': self.running,
            'input_url': self.input_url,
            'output_url': self.output_url,
            'width': self.width,
            'height': self.height,
            'framerate': self.framerate,
            'bitrate': self.bitrate,
            'output_mode': self.output_mode,
            'stats': self.stats.copy(),
            'queue_sizes': {
                'input': self.frame_queue.qsize(),
                'output': self.output_queue.qsize()
            }
        }
        
        return status
