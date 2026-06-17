import math
import os
import time

import cv2
from core.utils import imread_unicode

import torchvision.transforms.functional as TF
import sys
# 运行时打补丁：将旧版名称直接映射到新版模块上，欺骗 basicsr
sys.modules['torchvision.transforms.functional_tensor'] = TF

# --- Real-ESRGAN Dependencies ---
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    REALESRGAN_AVAILABLE = True
    print("Real-ESRGAN libraries loaded successfully.")
except ImportError as e:
    print(f"致命警告：无法加载超分模块！真实的 ImportError 原因如下:\n {e}")
    RealESRGANer, RRDBNet, SRVGGNetCompact = None, None, None
    REALESRGAN_AVAILABLE = False
except Exception as e:
    print(f"Unknown error importing Real-ESRGAN dependencies: {e}")
    RealESRGANer, RRDBNet, SRVGGNetCompact = None, None, None
    REALESRGAN_AVAILABLE = False

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
try:
    import torch

    TORCH_AVAILABLE = True
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(
            "\nWarning: MPS device support is preliminary. SAM2 was trained with CUDA; numerical deviations or performance drops may occur on MPS.")
    else:
        device = torch.device("cpu")
    print(f"Using execution device: {device}")

    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:  # Ampere architecture and newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 support enabled for Ampere GPU.")
except ImportError:
    print("Error: 'PyTorch' library not found. PyTorch is a core dependency of SAM2 and Real-ESRGAN.")
    torch = None
    TORCH_AVAILABLE = False
    device = torch.device("cpu")  # Fallback to CPU execution
except Exception as e:
    print(f"Unknown error initializing or validating PyTorch: {e}")
    torch = None
    TORCH_AVAILABLE = False
    device = torch.device("cpu")  # Fallback to CPU execution


# --- Real-ESRGAN Function ---
def enhance_image_impl(
        input_path, output_folder='results', model_name='RealESRGAN_x4plus',
        model_path='RealESRGAN_x4plus.pth', denoise_strength=0.5, outscale=4,
        suffix='out', tile=0, tile_pad=10, pre_pad=0, face_enhance=False,
        fp32=False, gpu_id=None, progress_callback=None):
    if not REALESRGAN_AVAILABLE or not TORCH_AVAILABLE or RRDBNet is None or RealESRGANer is None:
        raise ImportError("Real-ESRGAN or PyTorch libraries are not available.")

    model_name = model_name.split('.')[0]
    netscale = 4
    model_instance = None

    if model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:
        model_instance = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model_instance = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    else:
        try:
            model_instance = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        except Exception as model_init_err:
            raise ValueError(f"Failed to initialize model '{model_name}'.") from model_init_err

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint file not found: {model_path}")

    if progress_callback:
        progress_callback(5, "正在初始化超清放大引擎...")

    img = imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read input image at '{input_path}'")

    if progress_callback:
        progress_callback(15, "正在拆分图像网格...")

    h, w = img.shape[:2]
    # 判断是否包含 Alpha 透明通道 (BGRA)
    has_alpha = (len(img.shape) == 3 and img.shape[2] == 4)

    # 智能分块防卡死
    if tile <= 0:
        longest_edge = max(h, w)
        if longest_edge > 2000:
            tile = 128
        elif longest_edge > 1080:
            tile = 256
        else:
            tile = 512

    # ==============================================================================
    # 【修复核心】：精准预判 Real-ESRGAN 的内部分块逻辑，推算真实的 Tile 数量
    # ==============================================================================
    step = tile - tile_pad * 2
    if step <= 0: step = 1  # 容错保护

    tiles_x = math.ceil((w - tile) / step) + 1 if w > tile else 1
    tiles_y = math.ceil((h - tile) / step) + 1 if h > tile else 1
    base_tiles = tiles_x * tiles_y

    # 如果带有透明通道，RGB 算一次，Alpha 算一次，总次数翻倍
    total_passes = base_tiles * 2 if has_alpha else base_tiles

    try:
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model_instance,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=gpu_id
        )
    except Exception as init_err:
        raise ImportError(f"RealESRGANer initialization failed: {init_err}") from init_err

    # ==============================================================================
    # 动态进度条 Hook：显示具体的“通道”和准确的分块数，绝不超量
    # ==============================================================================
    hook_handle = None
    if progress_callback and hasattr(model_instance, 'register_forward_pre_hook'):
        state = {'current_pass': 0}

        def dynamic_progress_hook(module, input_data):
            state['current_pass'] += 1
            curr = state['current_pass']

            # 计算总进度 (20% ~ 85%)
            safe_total = max(total_passes, curr)
            ratio = min(curr / safe_total, 1.0)
            pct = 20 + int(65 * ratio)

            # 智能判断当前是处理 RGB 还是 Alpha 透明层
            if has_alpha and curr > base_tiles:
                phase_name = "透明通道"
                local_curr = curr - base_tiles
            else:
                phase_name = "色彩通道"
                local_curr = curr

            # 防止因边界计算误差导致显示 25/24
            local_curr = min(local_curr, base_tiles)

            progress_callback(pct, f"AI 正在处理{phase_name} (分块 {local_curr}/{base_tiles}) ...")

            # 释放 10ms 锁，保证 UI 不假死
            time.sleep(0.01)

        hook_handle = model_instance.register_forward_pre_hook(dynamic_progress_hook)
    # ==============================================================================

    try:
        if progress_callback:
            progress_callback(20, "开始 GPU 并行加速计算...")

        output, _ = upsampler.enhance(img, outscale=outscale)

    except RuntimeError as error:
        if 'out of memory' in str(error).lower():
            print("Notice: 显存不足，尝试自动缩减分块。")
        raise error
    finally:
        if hook_handle is not None:
            hook_handle.remove()

        if 'upsampler' in locals():
            del upsampler

        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    if progress_callback:
        progress_callback(95, "正在编码并合并输出结果...")

    return output