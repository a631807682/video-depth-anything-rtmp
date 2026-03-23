# Video Depth Anything - 实时视频2D转3D流处理器

基于Video Depth Anything深度学习模型的RTMP流2D转3D实时处理系统。

## 架构设计

```
RTMP流 → FFmpeg解码 → Video Depth Anything深度估计 → 3D转换 → FFmpeg编码 → RTMP输出
```

## 功能特性

### 输入流支持
- ✅ RTMP直播流
- ✅ RTSP流
- ✅ 文件输入流
- ✅ 本地摄像头

### 输出格式
- ✅ 红蓝3D (Anaglyph)
- ✅ 左右格式 (Side-by-Side)
- ✅ 深度图 (Depth Map)

### 技术特点
- 🚀 实时流处理
- 📱 支持多种设备 (CPU/GPU/MPS)
- 📊 详细的性能监控
- 🌐 Web界面控制
- 📦 Docker支持

## 环境要求

### 硬件要求
- **CPU**: Intel i5或同等级别
- **GPU**: NVIDIA CUDA支持或Apple Silicon
- **内存**: 8GB+
- **磁盘**: 10GB+

### 软件要求
- Python 3.8+
- FFmpeg 4.2+
- Git
- NVIDIA CUDA (可选，用于加速)

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/your/repo.git
cd video_depth_anything_3d
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或在Windows上:
# venv\Scripts\activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 下载预训练模型
```bash
mkdir -p pretrained
cd pretrained
# 下载Video Depth Anything预训练权重
wget https://github.com/bytedance/DeVideo3D/releases/download/v1.0/depth_anything_vitl14.pth
cd ..
```

### 5. 启动流处理

#### 使用命令行
```bash
python main.py \
  -i rtmp://your-source-stream \
  -o rtmp://your-destination-stream \
  -w 1920 -h 1080 -r 30 \
  -b 4M -m anaglyph -d auto -v
```

#### 使用Web界面
```bash
python web_server.py 8080
# 访问 http://localhost:8080
```

## 配置参数

### 输入参数
```
-i, --input         输入RTMP/RTSP流地址
-o, --output        输出RTMP流地址
```

### 视频参数
```
-w, --width         输出宽度 (默认: 1280)
-h, --height        输出高度 (默认: 720)
-r, --framerate     帧率 (默认: 30)
-b, --bitrate       比特率 (默认: 3M)
```

### 处理参数
```
-m, --mode          输出模式:
                    - anaglyph: 红蓝3D
                    - sbs: 左右格式
                    - depth: 深度图
-d, --device        处理设备:
                    - auto: 自动选择
                    - cpu: CPU
                    - cuda: NVIDIA CUDA
                    - mps: Apple Silicon
```

## Web界面使用

1. 启动Web服务器: `python web_server.py`
2. 访问: http://localhost:5000
3. 在界面中配置:
   - 输入流地址
   - 输出流地址
   - 视频参数
   - 输出模式
   - 设备选择

4. 点击"启动流处理"

## Docker部署

### 构建镜像
```bash
docker build -t video-depth-3d .
```

### 运行容器
```bash
docker run -d \
  --gpus all \
  -p 1935:1935 \
  -p 5000:5000 \
  -v $(pwd)/pretrained:/app/pretrained \
  --name video-depth-3d \
  video-depth-3d
```

## 性能优化

### 硬件加速
```bash
# NVIDIA CUDA
python main.py -i ... -o ... -d cuda

# Apple Silicon
python main.py -i ... -o ... -d mps
```

### 降低延迟
```bash
# 降低分辨率
python main.py -i ... -o ... -w 1280 -h 720

# 降低帧率
python main.py -i ... -o ... -r 25

# 优化FFmpeg参数
# 修改 ffmpeg_stream_processor.py 中的 preset 为 ultrafast
```

## 系统架构

### 模块说明

#### FFmpegStreamProcessor
负责与FFmpeg的输入输出流接口
- 从RTMP流读取原始帧
- 将处理后的帧发送到输出RTMP流
- 管理流状态和错误处理

#### VideoDepthAnything
深度学习深度估计模型
- 基于ResNet50的编码器
- 轻量级解码器
- 实时深度估计

#### 3D转换
- **红蓝3D**: 利用深度图偏移红色通道
- **左右格式**: 生成视差调整后的左右视图
- **深度图**: 可视化深度信息

## 性能监控

运行时会自动监控:
- 总帧数
- 处理帧数
- 丢弃帧数
- 平均处理时间
- 队列长度

## 常见问题

### 1. FFmpeg命令失败
```bash
# 检查FFmpeg安装
ffmpeg -version

# 如果没有安装，安装FFmpeg
# Ubuntu/Debian
sudo apt install ffmpeg -y

# macOS
brew install ffmpeg
```

### 2. 找不到模型文件
```bash
# 确保模型文件在正确位置
ls -la pretrained/
# 应该包含 depth_anything_vitl14.pth
```

### 3. 流延迟大
- 降低分辨率或帧率
- 使用GPU加速
- 减少FFmpeg缓冲区大小

### 4. 内存泄漏
- 检查PyTorch显存释放
- 增加系统内存
- 降低批量处理大小

## 扩展功能

### 添加新的输出格式
```python
# 在 video_depth_anything.py 中添加新的转换函数
def create_top_bottom(frame, depth):
    # 上下格式实现
    ...
    return top_bottom_image
```

### 自定义深度估计模型
```python
# 继承自 DepthAnything
class MyCustomModel(DepthAnything):
    def __init__(self, custom_params):
        super().__init__(pretrained=False)
        # 自定义模型结构
```

## 技术支持

### 提交问题
1. 确保使用最新版本
2. 提供:
   - 系统信息
   - 命令输出
   - 配置参数
   - 问题描述

### 参与开发
欢迎提交PR和Issue!

## 许可证

MIT License

## 贡献者

- 开发团队
- 社区贡献者

---

**提示**: 对于生产环境部署，建议使用Docker和监控工具。
