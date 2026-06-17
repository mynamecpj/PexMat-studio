import os
import sys
import ctypes
import traceback
import time
import multiprocessing
from typing import Optional

import cv2
import numpy as np
import torch

try:
    import psutil
except ImportError:
    psutil = None

from PySide6.QtGui import QPixmap, QBrush, QImage, Qt, QPainter
from PySide6.QtCore import QSize, QRect, QRectF, QPointF, QSizeF

# Settings and configurations
from config.settings import PREVIEW_BG_CHECKER_SIZE, PREVIEW_BG_COLOR1, PREVIEW_BG_COLOR2

# MatAnyone2 model imports (handled with fallback exceptions for compatibility)
try:
    from core.models.matanyone2.inference.inference_core import InferenceCore
    from core.models.matanyone2.matanyone2_wrapper import matanyone2
    import core.models.matanyone2.utils.device as mat_device
except ImportError:
    InferenceCore = None
    matanyone2 = None
    mat_device = None


def native_guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    Pure NumPy implementation of the Guided Filter algorithm.
    Serves as a reliable fallback when cv2.ximgproc is unavailable.

    Args:
        I (np.ndarray): Guidance image, can be single-channel or multi-channel.
        p (np.ndarray): Input image to be filtered (typically a coarse alpha mask).
        r (int): Local window radius.
        eps (float): Regularization parameter.

    Returns:
        np.ndarray: Filtered output mapped to 0.0 ~ 1.0.
    """
    # Convert multi-channel images to grayscale for guidance
    if len(I.shape) == 3:
        if I.shape[2] == 4:
            I_gray = cv2.cvtColor(I, cv2.COLOR_BGRA2GRAY)
        elif I.shape[2] == 3:
            I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        else:
            I_gray = I[:, :, 0]
    else:
        I_gray = I

    # Normalize guidance image to 0.0 ~ 1.0 range
    if I_gray.dtype == np.uint8:
        I_gray = I_gray.astype(np.float32) / 255.0

    # Vectorized calculations via fast box filtering
    mean_I = cv2.boxFilter(I_gray, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I_gray * p, -1, (r, r))

    # Covariance
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I_gray * I_gray, -1, (r, r))
    var_I = mean_II - mean_I * mean_I

    # Calculate linear coefficients a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Apply local averaging
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))

    q = mean_a * I_gray + mean_b
    return np.clip(q, 0.0, 1.0)


def upscale_mask_with_guidance(low_res_mask: np.ndarray, high_res_guide: np.ndarray,
                               subject_type: str = "product") -> np.ndarray:
    """
    Adaptive edge-preserving guided filter. Automatically tunes hyper-parameters
    based on object features (e.g., fine hair versus rigid product boundaries)
    to output high-fidelity refined masks at original resolution.
    """
    h_orig, w_orig = high_res_guide.shape[:2]

    # Resize coarse mask to original size using bilinear interpolation
    upsampled_mask = cv2.resize(low_res_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Adjust parameters dynamically based on structural elements
    if subject_type == "human_pet":
        # Hair/Fur: Small window and higher sensitivity to preserve fine details
        radius = 4
        eps = 1e-4
    else:
        # Rigid products/Objects: Larger window and stronger regularization to suppress edge halo artifacts
        radius = 8
        eps = 1e-2

    # Attempt to use native OpenCV contrib implementations for acceleration
    try:
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
            if len(high_res_guide.shape) == 3 and high_res_guide.shape[2] == 4:
                guide_image = cv2.cvtColor(high_res_guide, cv2.COLOR_BGRA2GRAY)
            elif len(high_res_guide.shape) == 3:
                guide_image = cv2.cvtColor(high_res_guide, cv2.COLOR_BGR2GRAY)
            else:
                guide_image = high_res_guide

            if guide_image.dtype == np.uint8:
                guide_image = guide_image.astype(np.float32) / 255.0

            refined_mask = cv2.ximgproc.guidedFilter(
                guide=guide_image, src=upsampled_mask.astype(np.float32),
                radius=radius, eps=eps, dDepth=-1
            )
            return np.clip(refined_mask, 0.0, 1.0)
    except Exception:
        pass

    # Fallback to pure NumPy guided filter implementation
    return native_guided_filter(high_res_guide, upsampled_mask.astype(np.float32), radius, eps)


# ==============================================================================
# --- Unicode Path Operations (OpenCV Compatibility) ---
# ==============================================================================

def imread_unicode(filename, flags=cv2.IMREAD_COLOR):
    """
    Safely reads images containing non-ASCII / Unicode paths.
    Uses context-managed stream opening to ensure files are released properly.
    """
    try:
        with open(filename, "rb") as stream:
            bytes_data = bytearray(stream.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpy_array, flags)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(filename, flags)


def imwrite_unicode(filename, img, params=None):
    """
    Safely writes images to paths containing Unicode characters.
    """
    try:
        ext = os.path.splitext(filename)[1]
        result, buf = cv2.imencode(ext, img, params)
        if result:
            with open(filename, "wb") as f:
                f.write(buf)
            return True
        else:
            return cv2.imwrite(filename, img, params)
    except Exception:
        return cv2.imwrite(filename, img, params)


def get_short_path_name_windows(long_name):
    """
    Converts a long filename on Windows platform into its 8.3 short format representation
    to prevent low-level read issues with non-ASCII or spaces in paths.
    """
    if os.name == 'nt':
        output_buf_size = 260
        output_buf = ctypes.create_unicode_buffer(output_buf_size)
        needed = ctypes.windll.kernel32.GetShortPathNameW(str(long_name), output_buf, output_buf_size)
        if needed == 0:
            return str(long_name)
        return output_buf.value
    return str(long_name)


# ==============================================================================
# --- NumPy Array to QPixmap Conversion Engine ---
# ==============================================================================

def convert_cv_to_pixmap(cv_image):
    """
    Converts an OpenCV NumPy NDArray to QPixmap safely.
    Includes safeguards against memory leaks and invalid format conversions.
    """
    if cv_image is None or cv_image.size == 0:
        return QPixmap()
    try:
        processed_img = cv_image

        # 1. Normalize floating-point matrices
        if np.issubdtype(cv_image.dtype, np.floating):
            processed_img = np.clip(cv_image * 255.0, 0, 255).astype(np.uint8)
        # 2. Normalize non-uint8 integer channels (e.g., uint16) to uint8
        elif cv_image.dtype != np.uint8:
            max_val = float(np.iinfo(cv_image.dtype).max) if np.issubdtype(cv_image.dtype, np.integer) else 255.0
            processed_img = np.clip((cv_image.astype(np.float32) / max_val) * 255.0, 0, 255).astype(np.uint8)

        # 3. Force C-contiguous memory layout
        if not processed_img.flags['C_CONTIGUOUS']:
            processed_img = np.ascontiguousarray(processed_img)

        height, width = processed_img.shape[:2]
        channels = processed_img.shape[2] if len(processed_img.shape) == 3 else 1

        # 4. Determine matching QImage format and byte offsets
        if channels == 1:
            qimage_format = QImage.Format.Format_Grayscale8
            bytes_per_line = width
        elif channels == 3:
            qimage_format = QImage.Format.Format_RGB888
            bytes_per_line = 3 * width
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            qimage_format = QImage.Format.Format_RGBA8888
            bytes_per_line = 4 * width
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGRA2RGBA)
        else:
            print(f"Unsupported number of channels: {channels}")
            return QPixmap()

        # 5. Build QImage structure
        q_img = QImage(processed_img.data, width, height, bytes_per_line, qimage_format)
        if q_img.isNull():
            return QPixmap()

        # 6. Perform a deep copy of the image to isolate memory from Python's GC tracking
        pixmap = QPixmap.fromImage(q_img).copy()
        return pixmap

    except cv2.error as e_cv:
        print(f"OpenCV error in convert_cv_to_pixmap: {e_cv}")
        traceback.print_exc()
        return QPixmap()
    except Exception as e_gen:
        print(f"Generic error in convert_cv_to_pixmap: {e_gen}")
        traceback.print_exc()
        return QPixmap()


def resize_image_to_max_dim(image, max_dim):
    """
    Rescales an image proportionally so its maximum dimension matches the specified limit.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image.copy(), 1.0

    if h > w:
        new_h = max_dim
        new_w = int(w * (new_h / h))
    else:
        new_w = max_dim
        new_h = int(h * (new_w / w))

    new_h = max(1, new_h)
    new_w = max(1, new_w)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scale_factor = max(h, w) / float(max_dim)
    return resized_image, scale_factor


# ==============================================================================
# --- Checkerboard Background Generator ---
# ==============================================================================

def create_checkerboard_pixmap(size: QSize, checker_size=PREVIEW_BG_CHECKER_SIZE):
    """
    Generates a high-performance checkered background pattern.
    Renders a micro 2x2 tile onto a QBrush texture and fills the space,
    avoiding heavy main-thread rendering loops.
    """
    if size.isEmpty():
        return QPixmap()

    tile_size = checker_size * 2
    tile = QImage(tile_size, tile_size, QImage.Format.Format_RGB32)

    color1 = PREVIEW_BG_COLOR1.rgb()
    color2 = PREVIEW_BG_COLOR2.rgb()

    for y in range(tile_size):
        for x in range(tile_size):
            use_color1 = ((x // checker_size) % 2 == (y // checker_size) % 2)
            tile.setPixel(x, y, color1 if use_color1 else color2)

    pix = QPixmap(size)
    painter = QPainter(pix)
    brush = QBrush(tile)
    painter.fillRect(QRect(0, 0, size.width(), size.height()), brush)
    painter.end()
    return pix


# ==============================================================================
# --- Thread Affinity and System Performance Optimizations ---
# ==============================================================================

def _optimize_cpu_core_affinity():
    """
    Binds CPU threads to physical cores (P-Cores), limits hyper-threading,
    and reserves at least 2 cores for the GUI main thread to prevent interface hanging.
    """
    try:
        if psutil is not None:
            p = psutil.Process(os.getpid())
            if sys.platform == "win32":
                p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            else:
                try:
                    p.nice(-5)
                except psutil.AccessDenied:
                    pass

            physical_cores = psutil.cpu_count(logical=False)
            if physical_cores is None:
                physical_cores = multiprocessing.cpu_count() // 2

            # Guard at least 2 cores to prevent GUI main thread starvation
            optimal_threads = max(1, physical_cores - 2)
            optimal_threads = min(optimal_threads, 10)  # Capped at 10 to balance overhead and performance

            os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
            os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_threads)

            try:
                torch.set_num_threads(optimal_threads)
            except Exception:
                pass

            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass

            print(f"[CPU Acceleration] Physical cores detected: {physical_cores} | Assigned worker threads: {optimal_threads} (Reserved 2 cores for GUI)")
        else:
            cores = max(1, (multiprocessing.cpu_count() // 2) - 1)
            try:
                torch.set_num_threads(cores)
            except Exception:
                pass
            print(f"[CPU Acceleration] psutil not installed. Falling back to thread limit: {cores}")

    except Exception as e:
        print(f"[CPU Acceleration] Initialization failed: {e}")


# ==============================================================================
# --- Isolated Inference Functions ---
# ==============================================================================

def _isolated_matanyone_inference(mat_model, roi_img_rgb, roi_mask_u8, device_str, erode_kernel_size,
                                  dilate_kernel_size, n_warmup=10) -> Optional[np.ndarray]:
    """
    取代原有混乱的 refine_mask_with_matanyone2，直接对接浮点精度的 Alpha 通道。
    """
    from core.models.matanyone2.inference.inference_core import InferenceCore
    from core.models.matanyone2.matanyone2_wrapper import matanyone2

    if mat_model is None or roi_img_rgb is None or roi_mask_u8 is None:
        return None

    t_start = time.time()
    device_obj = torch.device(device_str)

    try:
        h, w = roi_img_rgb.shape[:2]

        # 空间 16 像素对齐（防 Transformer 下采样丢失边缘）
        pad_h = (16 - (h % 16)) % 16
        pad_w = (16 - (w % 16)) % 16
        if pad_h > 0 or pad_w > 0:
            roi_img_padded = cv2.copyMakeBorder(roi_img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask_single = roi_mask_u8[:, :, 0] if roi_mask_u8.ndim == 3 else roi_mask_u8
            roi_mask_padded = cv2.copyMakeBorder(mask_single, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        else:
            roi_img_padded = roi_img_rgb
            roi_mask_padded = roi_mask_u8[:, :, 0] if roi_mask_u8.ndim == 3 else roi_mask_u8

        # 构建符合 MatAnyone 严格要求的绝对二值边界 (只存在 0 和 255)
        final_input_mask = np.zeros_like(roi_mask_padded, dtype=np.float32)
        final_input_mask[roi_mask_padded > 127] = 255.0

        if len(np.unique(final_input_mask)) == 1:
            final_input_mask[0, 0] = 255.0 if final_input_mask[0, 0] == 0.0 else 0.0

        with torch.inference_mode():
            processor = InferenceCore(mat_model, cfg=mat_model.cfg)
            processor.device = device_obj

            frames = [roi_img_padded, roi_img_padded]

            if device_obj.type == 'cuda':
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    foreground, alpha_out = matanyone2(
                        processor, frames, final_input_mask,
                        r_erode=erode_kernel_size, r_dilate=dilate_kernel_size, n_warmup=n_warmup
                    )
                torch.cuda.synchronize(device=device_obj)
            else:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                    foreground, alpha_out = matanyone2(
                        processor, frames, final_input_mask,
                        r_erode=erode_kernel_size, r_dilate=dilate_kernel_size, n_warmup=n_warmup
                    )

        # 接收并剥离最后一帧的高精度 float32 蒙版
        raw_alpha = alpha_out[-1]

        if raw_alpha.ndim == 3:
            raw_alpha = raw_alpha[:, :, 0]

        # 裁剪掉之前的 16 像素 padding
        final_alpha_cropped = raw_alpha[:h, :w]

        if hasattr(processor, 'clear_memory'):
            processor.clear_memory()

        print(f"[MatAnyone 2] Refined successfully in {time.time() - t_start:.4f}s")

        # 始终返回 float32 (0.0~1.0) 矩阵，交由 UI 层的 OpenCV/PySide6 画笔系统渲染
        return np.array(final_alpha_cropped, copy=True, order='C')

    except Exception as e:
        print(f"[MatAnyone 2] Inference failed: {e}")
        traceback.print_exc()
        return None


def _static_apply_mask_refinements(mask_np, image_np, refine_params, mat_model=None, device_str="cpu"):
    """
    重构后的后处理应用函数（去除了破坏透明度的硬边缘覆盖）
    """
    if mask_np is None:
        return None

    try:
        if refine_params.get('refine_matteformer_enabled', False) and mat_model is not None:
            if np.issubdtype(mask_np.dtype, np.floating):
                mask_raw_u8 = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
            else:
                mask_raw_u8 = (mask_np > 0).astype(np.uint8) * 255

            smooth = refine_params.get('refine_smooth', 0)
            feather = refine_params.get('refine_feather', 0)
            r_erode = max(10, int(smooth))
            r_dilate = max(10, int(feather))

            # 启用充分的 n_warmup=10 进行边缘细化
            refined_u8 = _isolated_matanyone_inference(
                mat_model, image_np, mask_raw_u8, device_str, r_erode, r_dilate, n_warmup=10
            )

            if refined_u8 is not None:
                # 移除了会导致发丝变成黑白锯齿硬块的 deterministic 边界保护逻辑
                # 保留 MatAnyone 2 原汁原味的软透明度发丝细节
                return refined_u8.astype(np.float32) / 255.0

        if mask_np.dtype == bool:
            refined = mask_np.astype(np.float32)
        else:
            refined = mask_np.astype(np.float32) / 255.0 if mask_np.max() > 1.0 else mask_np.astype(np.float32)

        # 降级处理
        shift = refine_params.get('refine_shift', 0)
        if shift != 0:
            kernel_size = abs(shift) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            if shift > 0:
                refined = cv2.dilate(refined, kernel)
            else:
                refined = cv2.erode(refined, kernel)

        smooth = refine_params.get('refine_smooth', 0)
        if smooth > 0:
            kernel_size = smooth * 2 + 1
            refined = cv2.GaussianBlur(refined, (kernel_size, kernel_size), 0)

        feather = refine_params.get('refine_feather', 0)
        if feather > 0:
            kernel_size = feather * 2 + 1
            refined = cv2.GaussianBlur(refined, (kernel_size, kernel_size), 0)

        return np.clip(refined, 0.0, 1.0)

    except Exception as e:
        traceback.print_exc()
        return None


# ==============================================================================
# --- General Utilities ---
# ==============================================================================

def get_asset_path(relative_path: str) -> str:
    """
    Returns the absolute path to files located within the package assets folder.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.getcwd()

    return os.path.join(base_path, "assets", relative_path)