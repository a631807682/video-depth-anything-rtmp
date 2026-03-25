# Video Depth Anything - 实时视频 2D 转 3D 流处理器

基于 Video Depth Anything 深度学习模型的 RTMP 流 2D 转 3D 实时处理系统。

## Quick Start

### 安装依赖

```bash
python -m venv venv3.12
source venv/bin/activate
pip install -r requirements.txt
```

### 命令执行

```bash
python main.py -i rtmp://127.0.0.1:1935/stream/2d -o rtmp://127.0.0.1:1935/stream/3d -r 30 -b 3M -m half-sbs -d auto
```

## 功能特性

### 输入流支持

- ✅ RTMP 直播流

### 输出格式

- ✅ Half Side-by-Side
- ✅ Side-by-Side

## 环境要求

### 硬件要求

- **GPU**: NVIDIA CUDA

### 软件要求

- Python 3.12+
- FFmpeg 4.2+
- Git
- NVIDIA CUDA
