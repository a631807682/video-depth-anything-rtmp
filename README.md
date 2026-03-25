# Video Depth Anything - 实时视频 2D 转 3D 流处理器

基于 Video Depth Anything 深度学习模型的 RTMP 流 2D 转 3D 实时处理系统。

## Quick Start

### 安装依赖

```bash
python -m venv venv3.12
source venv/bin/activate
pip install -r requirements.txt

# 模型下载
https://huggingface.co/depth-anything/Depth-Anything-V2-Small
```

### 命令执行

```bash
python main.py -i rtmp://127.0.0.1:1935/stream/2d -o rtmp://127.0.0.1:1935/stream/3d -r 30 -b 3M -m half-sbs -d auto
```

### TensorRT 加速

#### Onnx

```bash
# Onnx 下载
https://huggingface.co/onnx-community/depth-anything-v2-small/
```

#### tensorrt

<!-- ```bash
# download tensorrt
## tensorrt
https://developer.nvidia.com/nvidia-tensorrt-8x-download
## cudnn
https://developer.nvidia.com/rdp/cudnn-archive

tar -xzvf TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.tar.gz

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/3dmodel/TensorRT-8.5.1.7/lib
/opt/TensorRT-8.5.1.7/bin/trtexec --version

tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-8.9.7.29_cuda11-archive
sudo cp -P lib/* /opt/3dmodel/TensorRT-8.5.1.7/lib/
# 拷贝头文件（如果以后需要编译代码的话）
sudo cp include/* /opt/3dmodel/TensorRT-8.5.1.7/include/
sudo chmod 755 /opt/3dmodel/TensorRT-8.5.1.7/lib/libcudnn*

trtexec --version

# 进入目录或指定全路径
trtexec --onnx=./pretrained/depth_anything_v2_vits_fp16.onnx \
        --saveEngine=./pretrained/depth_vits_fp16.engine \
        --fp16 \
        --minShapes=pixel_values:1x3x518x518 \
        --optShapes=pixel_values:1x3x518x518 \
        --maxShapes=pixel_values:1x3x518x518
``` -->

```bash
# TensorRT + CuPy
pip install nvidia-tensorrt cupy-cuda12x

# Engine 与 python 环境强相关，需要自行生成
python engine_export.py
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
