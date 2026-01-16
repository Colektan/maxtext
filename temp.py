import sys
import platform
import subprocess
import os

def get_cuda_version():
    # 尝试通过 nvcc 获取
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        for line in output.split('\n'):
            if "release" in line:
                return line.strip()
    except:
        pass
    
    # 尝试通过 nvidia-smi 获取
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        for line in output.split('\n'):
            if "CUDA Version" in line:
                # 格式通常是: ... CUDA Version: 11.x ...
                parts = line.split("CUDA Version:")
                if len(parts) > 1:
                    return parts[1].split()[0].strip()
    except:
        pass

    # 尝试读取文件
    if os.path.exists("/usr/local/cuda/version.txt"):
        with open("/usr/local/cuda/version.txt") as f:
            return f.read().strip()
            
    return "未找到 (Not found)"

def get_cudnn_version():
    # 常见的头文件路径
    paths = [
        "/usr/include/cudnn_version.h",
        "/usr/local/cuda/include/cudnn_version.h",
        "/usr/include/cudnn.h",
        "/usr/local/cuda/include/cudnn.h"
    ]
    
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                content = f.read()
                major = None
                minor = None
                patch = None
                for line in content.split('\n'):
                    if "#define CUDNN_MAJOR" in line:
                        major = line.split()[-1]
                    if "#define CUDNN_MINOR" in line:
                        minor = line.split()[-1]
                    if "#define CUDNN_PATCHLEVEL" in line:
                        patch = line.split()[-1]
                
                if major and minor:
                    return f"{major}.{minor}.{patch}" if patch else f"{major}.{minor}"
    return "未找到 (Not found)"

print("="*30)
print(f"Python Version: {platform.python_version()} (cp{sys.version_info.major}{sys.version_info.minor})")
print(f"OS Platform:    {platform.platform()}")
print(f"CUDA Version:   {get_cuda_version()}")
print(f"cuDNN Version:  {get_cudnn_version()}")

try:
    import jax
    print(f"JAX Version:    {jax.__version__}")
except ImportError:
    print("JAX Version:    未安装 (Not installed)")
print("="*30)