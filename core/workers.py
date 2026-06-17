import gc
import os
import sys
import tempfile
import time
import math
import traceback
import collections
import av
from contextlib import nullcontext
import uuid
from fractions import Fraction
import numpy as np
import cv2
from PIL import Image as PILImage
from PIL import ImageSequence


import platform
import subprocess
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from core.utils import _static_apply_mask_refinements, upscale_mask_with_guidance

import torch
from pydub import AudioSegment
from torchvision import transforms
from torchvision.transforms import functional as F
from transformers.models.auto.modeling_auto import AutoModelForImageSegmentation
from ultralytics import YOLO
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from hydra.core.global_hydra import GlobalHydra

from PySide6.QtCore import (
    QObject, Signal, QThread, Slot, QPointF, QRectF, QSizeF, Qt, QSettings
)
from PySide6.QtGui import QPixmap, QImage, QTransform, QColor

# --- App Core & Configuration Mappings ---
from core.models.matanyone2.matanyone2_wrapper import matanyone2
try:
    from core.models.matanyone2.inference.inference_core import InferenceCore
    from core.models.matanyone2.utils.get_default_model import get_matanyone2_model
    import core.models.matanyone2.utils.device as mat_device
except ImportError:
    InferenceCore = None
    get_matanyone2_model = None
    mat_device = None

from core.utils import (
    imread_unicode, convert_cv_to_pixmap, imwrite_unicode,
    get_short_path_name_windows, _static_apply_mask_refinements,
    _optimize_cpu_core_affinity, _isolated_matanyone_inference
)
from config.settings import (
    SAM2_IMAGE_CHECKPOINT_PATH, SAM2_IMAGE_MODEL_CFG_PATH,
    SAM2_VIDEO_CHECKPOINT_PATH, SAM2_VIDEO_MODEL_CFG_PATH,
    VIDEO_DEFAULT_FPS, VIDEO_FRAME_EXT, VIDEO_THUMBNAIL_EXT, VIDEO_PLAYBACK_INTERVAL_MS,
    MATANYONE_CHECKPOINT_PATH, _TR, TEMP_BASE_DIR
)
from core.models.realesrgan_root import enhance_image_impl, REALESRGAN_AVAILABLE
from ui.views.video_view import VideoThumbnailScrubber

# --- SAM2 Framework Builders ---
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

# --- Environment Configurations ---
PILLOW_AVAILABLE = True
TORCH_AVAILABLE = True
SAM2_IMAGE_PREDICTOR_AVAILABLE = True
SAM2_VIDEO_PREDICTOR_AVAILABLE = True
SAM2_AVAILABLE = True

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# --- Target Acceleration Hardware ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:  # Ampere architecture or newer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ==============================================================================
# --- Worker Base & Core Loaders ---
# ==============================================================================

class WorkerBase(QObject):
    """
    Abstract base worker implementation containing standard signaling pipelines.
    """
    finished = Signal(object, bool, str)
    progress = Signal(int, str)
    error = Signal(str)
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_cancelled = False

    @Slot()
    def cancel(self):
        self.log_message.emit("Worker thread cancellation requested.")
        self._is_cancelled = True


class VideoInitStateWorker(WorkerBase):
    """
    深度优化的 SAM2 视频状态初始化工作线程。
    通过系统物理内存卸载 (Offloading) 与异步流式懒加载 (Async Loading) 技术，彻底解决超长视频初始化崩溃和爆显存的问题。
    """

    def run_init(self, video_predictor, temp_frame_dir):
        inference_state = None
        success = False
        error_msg = ""

        try:
            import threading
            import torch
            from PySide6.QtCore import QThread

            real_target_device = PredictWorker._get_true_target_device()
            device_obj = torch.device(real_target_device)

            if device_obj.type == 'cuda':
                torch.zeros(1).to(device_obj)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True

            self.progress.emit(0, "正在准备视频流特征缓存...")
            QThread.msleep(50)

            # 进程状态共享字典，提供跨线程的高频 GUI 渲染和防假死刷新
            shared_state = {
                "current": 0,
                "total": 0,
                "phase": 1,  # 1: 物理帧解析解码, 2: 神经网络深度编码对齐
                "error": None,
                "state": None
            }

            def _heavy_init():
                try:
                    def on_load_progress(current, total):
                        shared_state["current"] = current
                        shared_state["total"] = total
                        if total > 0 and current >= total - 1:
                            shared_state["phase"] = 2

                    with torch.inference_mode():
                        # 开启 CPU 卸载和异步帧流懒加载
                        state = video_predictor.init_state(
                            video_path=temp_frame_dir,
                            offload_video_to_cpu=True,   # 基础大帧图像放在系统物理内存
                            offload_state_to_cpu=True,   # 历史时序特征卸载到系统物理内存
                            async_loading_frames=True,   # 开启异步帧懒加载，避免瞬间暴涨
                            progress_callback=on_load_progress
                        )
                    shared_state["state"] = state
                except Exception as e:
                    shared_state["error"] = e

            # 将重度阻塞计算丢到底层工作线程
            worker_thread = threading.Thread(target=_heavy_init)
            worker_thread.start()

            fake_pct = 90
            # GUI 心跳主线程：拦截未响应状态，保持进度平滑流畅，并在显卡计算期间持续释放 CPU 挂起
            while worker_thread.is_alive():
                if self._is_cancelled:
                    break

                phase = shared_state["phase"]
                current = shared_state["current"]
                total = shared_state["total"]

                if phase == 1 and total > 0:
                    pct = int((current / total) * 90)
                    self.progress.emit(pct, f"流式解析视频序列: {current}/{total} 帧")
                    QThread.msleep(15)
                elif phase == 2:
                    if fake_pct < 99:
                        fake_pct += 1
                    self.progress.emit(fake_pct, f"时序神经网络特征深度对齐中 ({fake_pct}%)...")
                    QThread.msleep(150)
                else:
                    QThread.msleep(15)

            worker_thread.join()

            if shared_state["error"] is not None:
                raise shared_state["error"]

            inference_state = shared_state["state"]
            self.progress.emit(100, "视频空间深度特征编码完毕！")
            QThread.msleep(100)
            success = True

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.finished.emit(inference_state, success, error_msg)


class VideoSyncHistoryWorker(WorkerBase):
    """
    Asynchronous history reconstruction thread. Recalculates historical layers
    and previous frame parameters without blocking main GUI thread execution.
    """
    sync_complete = Signal(dict, bool, str)

    def run_sync(self, video_predictor, inference_state, target_points, start_frame):
        success = False
        error_msg = ""
        temp_multi_masks = {}

        try:
            real_target_device = PredictWorker._get_true_target_device()
            use_autocast = (real_target_device == 'cuda')

            # 强制使用标准的 float16，彻底杜绝 BFloat16 与 Float32 冲突造成的线性投影崩盘
            autocast_dtype = torch.float16
            autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else nullcontext()

            with torch.inference_mode(), autocast_ctx:
                video_predictor.reset_state(inference_state)

                total_targets = len(target_points)
                for idx, (obj_id, target_data) in enumerate(target_points.items()):
                    # 基于 1 进制提示图层同步进度
                    self.progress.emit(
                        int((idx / max(1, total_targets)) * 100),
                        f"Reconstructing layer target object {obj_id + 1}..."
                    )

                    frame_idx = target_data.get('annotation_frame')
                    if frame_idx is not None and (target_data.get('points') or target_data.get('box')):
                        input_points_np, input_labels_np, box_np = None, None, None

                        if target_data.get('points'):
                            pts = target_data['points']
                            input_points_np = np.array([(p[0], p[1]) for p in pts], dtype=np.float32)
                            if input_points_np.ndim == 1 and input_points_np.size > 0:
                                input_points_np = input_points_np[np.newaxis, :]
                            input_labels_np = np.array([p[2] for p in pts], dtype=np.int32)

                        if target_data.get('box'):
                            box_np = np.array(target_data['box'], dtype=np.float32)

                        local_ann_frame = frame_idx - start_frame

                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=local_ann_frame, obj_id=obj_id,
                            points=input_points_np, labels=input_labels_np, box=box_np,
                            clear_old_points=True
                        )

                        if out_mask_logits is not None and len(out_mask_logits) > 0 and out_obj_ids is not None:
                            for i, obj_id_tensor in enumerate(out_obj_ids):
                                o_id_val = int(obj_id_tensor.item()) if hasattr(obj_id_tensor, 'item') else int(obj_id_tensor)
                                mask = (out_mask_logits[i].float() > 0.0).cpu().numpy().squeeze()
                                if mask.dtype != bool:
                                    mask = mask.astype(bool)
                                mask = np.ascontiguousarray(mask)

                                if frame_idx not in temp_multi_masks:
                                    temp_multi_masks[frame_idx] = {}
                                temp_multi_masks[frame_idx][o_id_val] = mask
            success = True
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
        finally:
            self.sync_complete.emit(temp_multi_masks, success, error_msg)


class VideoInteractionWorker(WorkerBase):
    """
    Asynchronous handler representing user interactive prompts on active video frames.
    Implements physical deep copies on output buffers to avoid cross-thread race conditions.
    """
    def run_interaction(self, video_predictor, inference_state, frame_idx, obj_id, points, labels, box, clear_old_points):
        success = False
        error_msg = ""
        result_masks = {}

        try:
            real_target_device = PredictWorker._get_true_target_device()
            use_autocast = (real_target_device == 'cuda')

            # 强制统一 Autocast 精度为标准 float16 格式
            autocast_dtype = torch.float16
            autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else nullcontext()

            with torch.inference_mode(), autocast_ctx:
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    box=box,
                    clear_old_points=clear_old_points
                )

                if out_mask_logits is not None and len(out_mask_logits) > 0 and out_obj_ids is not None:
                    for i, obj_id_tensor in enumerate(out_obj_ids):
                        obj_id_val = int(obj_id_tensor.item()) if hasattr(obj_id_tensor, 'item') else int(obj_id_tensor)
                        mask = (out_mask_logits[i].float() > 0.0).cpu().numpy().squeeze()
                        if mask.dtype != bool:
                            mask = mask.astype(bool)

                        result_masks[obj_id_val] = np.array(mask, copy=True, order='C')

            success = True
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
        finally:
            self.finished.emit(result_masks, success, error_msg)


class HeadlessLoader(QObject):
    """
    Invisible backend module loader. Syncs models sequentially
    (SAM2, MatAnyone2, YOLO, BiRefNet, MEMatte) on isolated threads.
    """
    loading_complete = Signal(object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded_image_predictor = None
        self._loaded_video_predictor = None
        self._loaded_matanyone_model = None

        self._thread = QThread(self)
        self.moveToThread(self._thread)
        self._thread.started.connect(self.run)

    def start(self):
        """Starts the asynchronous loader thread."""
        self._thread.start()

    def run(self):
        """Sequentially loads required weights on thread execution."""
        # 1. Load SAM2 Image Predictor
        img_predictor, success, _ = self._load_model_sync("image")
        if success:
            self._loaded_image_predictor = img_predictor

        # 2. Load SAM2 Video Predictor
        vid_predictor, success, _ = self._load_model_sync("video")
        if success:
            self._loaded_video_predictor = vid_predictor

        # 3. Load MatAnyone 2 Model
        mat_model, mat_success, mat_err = self._load_matanyone_sync()
        if mat_success:
            self._loaded_matanyone_model = mat_model

        # 4. Silent-preload localized segmentation networks
        self._preload_auto_matting_models()

        # 5. Emit completed signal and release thread execution context
        self.loading_complete.emit(
            self._loaded_image_predictor,
            self._loaded_video_predictor,
            self._loaded_matanyone_model
        )
        self._thread.quit()

    def _preload_auto_matting_models(self):
        """Preloads segmentation models in the background to ensure immediate interface responses."""
        try:
            app_path = sys._MEIPASS if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS') else os.getcwd()
            target_device = PredictWorker._get_true_target_device()

            if target_device == "cpu":
                _optimize_cpu_core_affinity()

            # A. Preload YOLO Model
            yolo_path = os.path.join(app_path, "checkpoints", "yolo26s.pt")
            if os.path.exists(yolo_path) and PredictWorker._cached_yolo_model is None:
                PredictWorker._cached_yolo_model = YOLO(yolo_path)
                PredictWorker._cached_yolo_model.to(target_device)

            # B. Preload BiRefNet Model
            birefnet_path = os.path.join(app_path, "checkpoints", "BiRefNet_local")
            if os.path.exists(birefnet_path) and PredictWorker._cached_birefnet_model is None:
                biref_model = AutoModelForImageSegmentation.from_pretrained(
                    birefnet_path, trust_remote_code=True, local_files_only=True, use_safetensors=True
                )
                if target_device == "cpu":
                    biref_model = biref_model.float()
                    biref_model = torch.ao.quantization.quantize_dynamic(biref_model, {torch.nn.Linear}, dtype=torch.qint8)
                biref_model.to(target_device).eval()
                PredictWorker._cached_birefnet_model = biref_model

            # C. Preload MEMatte Model
            cfg_path = os.path.join(app_path, "MEMatte", "configs", "MEMatte_B_topk0.25_win_global_long.py")
            ckpt_path = os.path.join(app_path, "checkpoints", "MEMatte_ViTB_DIM.pth")

            if os.path.exists(cfg_path) and os.path.exists(ckpt_path) and PredictWorker._cached_mematte_model is None:
                cfg = LazyConfig.load(cfg_path)
                cfg.model.teacher_backbone = None
                cfg.model.backbone.max_number_token = 18500
                mematte_model = instantiate(cfg.model)

                if target_device == "cpu":
                    mematte_model = mematte_model.float()

                # Weights must be loaded prior to dynamic quantization
                DetectionCheckpointer(mematte_model).load(ckpt_path)

                if target_device == "cpu":
                    mematte_model = torch.ao.quantization.quantize_dynamic(mematte_model, {torch.nn.Linear}, dtype=torch.qint8)

                mematte_model.to(target_device).eval()
                PredictWorker._cached_mematte_model = mematte_model

            if target_device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Background preloading warning: {e}")

    def _load_model_sync(self, predictor_type: str) -> tuple[object | None, bool, str]:
        """Synchronous loader structure targeting SAM 2 components."""
        try:
            true_device_str = PredictWorker._get_true_target_device()
            device_obj = torch.device(true_device_str)

            if predictor_type == "image":
                model_path, cfg_path = SAM2_IMAGE_CHECKPOINT_PATH, SAM2_IMAGE_MODEL_CFG_PATH
            else:
                model_path, cfg_path = SAM2_VIDEO_CHECKPOINT_PATH, SAM2_VIDEO_MODEL_CFG_PATH

            app_path = sys._MEIPASS if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS') else os.getcwd()
            resolved_model_path = os.path.join(app_path, model_path)
            resolved_cfg_path = os.path.join(app_path, cfg_path)

            if not os.path.exists(resolved_model_path):
                resolved_model_path = model_path
            if not os.path.exists(resolved_cfg_path):
                resolved_cfg_path = cfg_path

            if not os.path.exists(resolved_model_path) or not os.path.exists(resolved_cfg_path):
                raise FileNotFoundError("Model files or configuration layout templates missing.")

            if predictor_type == "image":
                sam2_model = build_sam2(resolved_cfg_path, resolved_model_path, device=device_obj)
                predictor_instance = SAM2ImagePredictor(sam2_model)
            else:
                predictor_instance = build_sam2_video_predictor(resolved_cfg_path, resolved_model_path, device=device_obj)

            return predictor_instance, True, ""
        except Exception as e:
            traceback.print_exc()
            return None, False, str(e)

    def _load_matanyone_sync(self) -> tuple[object | None, bool, str]:
        """Synchronous loader structure targeting MatAnyone2 pipeline."""
        try:
            true_device_str = PredictWorker._get_true_target_device()
            device_obj = torch.device(true_device_str)

            app_path = sys._MEIPASS if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS') else os.getcwd()
            resolved_path = os.path.join(app_path, MATANYONE_CHECKPOINT_PATH)

            if not os.path.exists(resolved_path):
                if os.path.exists(MATANYONE_CHECKPOINT_PATH):
                    resolved_path = MATANYONE_CHECKPOINT_PATH
                else:
                    return None, False, f"MatAnyone2 model checkpoint not found: {MATANYONE_CHECKPOINT_PATH}"

            try:
                GlobalHydra.instance().clear()
            except Exception:
                pass

            mat_model = get_matanyone2_model(resolved_path, true_device_str)
            mat_model = mat_model.to(device_obj).eval()

            return mat_model, True, ""
        except Exception as e:
            traceback.print_exc()
            return None, False, f"MatAnyone2 load error: {str(e)}"


# ==============================================================================
# --- Task Specific Workers ---
# ==============================================================================

class ImageLoaderWorker(QObject):
    """
    Worker module handling the background decoding and parsing of image resources.
    """
    image_loaded = Signal(QPixmap, str, str, str)

    def __init__(self, file_path, item_id, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.item_id = item_id
        self._is_cancelled = False

    @Slot()
    def run(self):
        error_msg = ""
        pixmap = QPixmap()
        try:
            if self._is_cancelled:
                raise InterruptedError("Image loading cancelled by user.")

            pixmap = QPixmap(self.file_path)

            if pixmap.isNull():
                cv_image = imread_unicode(self.file_path, cv2.IMREAD_UNCHANGED)
                if cv_image is not None:
                    pixmap = convert_cv_to_pixmap(cv_image)

            if pixmap.isNull():
                error_msg = f"Failed to load image file: {os.path.basename(self.file_path)}"

        except InterruptedError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"Error loading image '{os.path.basename(self.file_path)}': {e}"
            traceback.print_exc()

        self.image_loaded.emit(pixmap, self.file_path, self.item_id, error_msg)

    def cancel(self):
        self._is_cancelled = True


class BlurWorker(QObject):
    """
    Worker tasked with constructing transition frames on arbitrary levels of Gaussian blur.
    """
    new_frame_ready = Signal(QPixmap)
    finished = Signal()

    def __init__(self, base_pixmap: QPixmap, max_sigma: int = 20, steps: int = 15, parent=None):
        super().__init__(parent)
        self._base_pixmap = base_pixmap
        self._max_sigma = max_sigma
        self._steps = steps
        self._is_cancelled = False

    @Slot()
    def run(self):
        try:
            if self._base_pixmap.isNull():
                self.finished.emit()
                return

            qimage = self._base_pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
            view = qimage.constBits()
            cv_image_bgra = np.array(view, copy=True).reshape(qimage.height(), qimage.width(), 4)

            for i in range(1, self._steps + 1):
                if self._is_cancelled:
                    break

                current_sigma = (i / self._steps) * self._max_sigma
                blurred_cv_image = cv2.GaussianBlur(cv_image_bgra, (0, 0), sigmaX=current_sigma, sigmaY=current_sigma)

                h, w, ch = blurred_cv_image.shape
                bytes_per_line = ch * w
                blurred_qimage = QImage(
                    np.ascontiguousarray(blurred_cv_image).data,
                    w, h, bytes_per_line,
                    QImage.Format.Format_ARGB32_Premultiplied
                )
                frame_pixmap = QPixmap.fromImage(blurred_qimage)

                if not frame_pixmap.isNull():
                    self.new_frame_ready.emit(frame_pixmap)

                QThread.msleep(20)

        except Exception as e:
            traceback.print_exc()
        finally:
            self.finished.emit()

    def cancel(self):
        self._is_cancelled = True


class ModelLoaderWorker(WorkerBase):
    """
    Explicit model loading worker pipeline.
    """
    @Slot(str, str, str, object)
    def run_load(self, predictor_type, model_path, cfg_path, device_obj):
        predictor_instance = None
        success = False
        error_msg = ""
        start_time = time.time()

        try:
            if not os.path.exists(model_path) or not os.path.exists(cfg_path):
                raise FileNotFoundError(f"SAM2 assets missing:\nM={model_path}\nC={cfg_path}")

            if predictor_type == "image":
                sam2_model = build_sam2(cfg_path, model_path, device=device_obj)
                predictor_instance = SAM2ImagePredictor(sam2_model)
            elif predictor_type == "video":
                predictor_instance = build_sam2_video_predictor(cfg_path, model_path, device=device_obj)
            else:
                raise ValueError(f"Unsupported predictor layout: {predictor_type}")

            success = True
            self.log_message.emit(f"SAM2 {predictor_type} predictor loaded successfully in {time.time() - start_time:.2f}s.")

        except Exception as e:
            error_msg = f"Failed to load SAM2 {predictor_type} model: {e}"
            self.log_message.emit(f"Error: {error_msg}")
            traceback.print_exc()
            predictor_instance = None
            success = False

        finally:
            if device_obj.type == 'cuda':
                torch.cuda.empty_cache()
            self.finished.emit(predictor_instance, success, error_msg)


class EnhanceWorker(WorkerBase):
    """
    Upscaling worker running Real-ESRGAN backend processing streams.
    """
    @Slot(str, str, float, int, object, int, int)
    def run_enhance(self, input_path, enhance_model_path, enhance_fixed_denoise, selected_outscale, device_obj,
                    tile_size, tile_pad):
        self.log_message.emit(
            f"Worker thread started: Image upscaling (Ratio: {selected_outscale}x, Tiles: {tile_size if tile_size > 0 else 'None/Auto'}, Pad: {tile_pad}px)..."
        )
        output_cv_image = None
        success = False
        error_msg = ""
        start_time = time.time()

        try:
            if not os.path.exists(enhance_model_path):
                raise FileNotFoundError(f"Enhancement checkpoints missing: {enhance_model_path}")

            enhance_gpu_id = None
            if device_obj.type == 'cuda' and torch.cuda.is_available():
                enhance_gpu_id = device_obj.index if device_obj.index is not None else 0

            model_name_base = os.path.splitext(os.path.basename(enhance_model_path))[0]

            def _progress_callback(percent, message):
                self.progress.emit(percent, message)

            if self._is_cancelled:
                raise InterruptedError("Upscaling operation cancelled by user.")

            output_cv_image = enhance_image_impl(
                input_path=input_path,
                model_name=model_name_base,
                model_path=enhance_model_path,
                denoise_strength=enhance_fixed_denoise,
                outscale=selected_outscale,
                fp32=(device_obj.type == 'cpu'),
                gpu_id=enhance_gpu_id,
                progress_callback=_progress_callback,
                tile=tile_size,
                tile_pad=tile_pad
            )

            if output_cv_image is None:
                raise RuntimeError("Upscaling failed to generate an output.")

            success = True
            self.log_message.emit(f"Upscaling completed successfully in {time.time() - start_time:.2f}s.")

        except InterruptedError:
            error_msg = "Upscaling operation cancelled by user."
            self.log_message.emit(error_msg)
            success = False
        except Exception as e:
            error_msg = f"Error during image upscaling: {e}"
            self.log_message.emit(f"Error: {error_msg}")
            traceback.print_exc()
            output_cv_image = None
            success = False
        finally:
            if device_obj.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self.finished.emit(output_cv_image, success, error_msg)


class PredictWorker(WorkerBase):
    """
    High-precision background inference thread for automated and prompt-based matting.
    Defines fallback strategies for standard products and human subjects.
    """
    _cached_yolo_model = None
    _cached_birefnet_model = None
    _cached_mematte_model = None
    _internal_start_signal = Signal(object, tuple, object, bool, object)

    @classmethod
    def clear_models_cache(cls):
        """Forces clearance of local preloaded models when hardware targets shift."""
        cls._cached_yolo_model = None
        if cls._cached_birefnet_model is not None:
            del cls._cached_birefnet_model
            cls._cached_birefnet_model = None
        if cls._cached_mematte_model is not None:
            del cls._cached_mematte_model
            cls._cached_mematte_model = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._internal_start_signal.connect(self._do_run_predict, Qt.ConnectionType.QueuedConnection)

    @staticmethod
    def _get_true_target_device():
        """Reads target configuration directly from local properties files."""
        app_name = "ImageVideoToolbox"
        if os.name == 'nt':
            path = os.path.join(os.environ.get('APPDATA', ''), app_name)
        else:
            path = os.path.join(os.path.expanduser('~'), '.config', app_name)
        settings_path = os.path.join(path, "settings.ini")

        if not os.path.exists(settings_path):
            return "cuda" if torch.cuda.is_available() else (
                "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
            )

        settings = QSettings(settings_path, QSettings.Format.IniFormat)
        use_gpu = settings.value("hardware/use_gpu", True, type=bool)

        if str(use_gpu).lower() in ['true', '1', 't', 'y']:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def fill_alpha_internal_holes(self, alpha_np, max_hole_area=30):
        """Fills tiny pinhole discrepancies within continuous segmentation masks."""
        _, binary_inv = cv2.threshold(alpha_np, 254, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        alpha_fixed = alpha_np.copy()
        if hierarchy is not None:
            for i in range(len(contours)):
                if hierarchy[0][i][3] != -1:
                    area = cv2.contourArea(contours[i])
                    if area < max_hole_area:
                        cv2.drawContours(alpha_fixed, [contours[i]], 0, 255, -1)
        return alpha_fixed

    def refine_alpha_natural_clarity(self, alpha_np, clip_low=20, clip_high=240):
        """Contrast stretches low/high bounds of alpha transition regions."""
        alpha_f = alpha_np.astype(np.float32)
        alpha_f = np.clip((alpha_f - clip_low) * (255.0 / (max(1, clip_high - clip_low))), 0, 255)
        return alpha_f.astype(np.uint8)

    def natural_hair_color_decontamination(self, image_np, alpha_np, erode_safe=10, intensity=3.0, hair_y_ratio=1.0):
        """Extrapolates interior solid colors to edge pixels to negate background color bleeding."""
        H, W = image_np.shape[:2]
        safe_mask = np.where(alpha_np >= 254, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_safe * 2 + 1, erode_safe * 2 + 1))
        core_alpha = cv2.erode(safe_mask, kernel, iterations=1)

        if cv2.countNonZero(core_alpha) == 0:
            _, core_alpha = cv2.threshold(alpha_np, 245, 255, cv2.THRESH_BINARY)

        core_mask_f = (core_alpha / 255.0).astype(np.float32)
        core_img = image_np.astype(np.float32) * core_mask_f[:, :, np.newaxis]

        blur_radius = int(max(H, W) * 0.08)
        if blur_radius % 2 == 0:
            blur_radius += 1
        blur_radius = max(51, blur_radius)

        blur_img = cv2.GaussianBlur(core_img, (blur_radius, blur_radius), 0)
        blur_mask = cv2.GaussianBlur(core_mask_f, (blur_radius, blur_radius), 0)

        if blur_mask.ndim == 2:
            blur_mask = blur_mask[:, :, np.newaxis]

        pure_extrapolated_colors = blur_img / (blur_mask + 1e-5)
        pure_extrapolated_colors = np.clip(pure_extrapolated_colors, 0, 255).astype(np.uint8)

        img_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        extr_lab = cv2.cvtColor(pure_extrapolated_colors, cv2.COLOR_RGB2LAB)

        result_lab = img_lab.copy()
        result_lab[:, :, 1] = extr_lab[:, :, 1]
        result_lab[:, :, 2] = extr_lab[:, :, 2]

        decontaminated_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

        alpha_f = alpha_np.astype(np.float32) / 255.0
        inv_alpha = np.clip(1.0 - alpha_f, 0.0, 1.0)
        blend_weight = np.clip(np.power(inv_alpha, 0.3) * intensity, 0.0, 1.0)
        blend_weight[alpha_np == 0] = 0

        spatial_mask = np.zeros((H, W), dtype=np.float32)
        target_h = int(H * hair_y_ratio)
        fade_h = int(H * 0.1)
        spatial_mask[:target_h, :] = 1.0

        for y in range(target_h, min(target_h + fade_h, H)):
            spatial_mask[y, :] = 1.0 - (y - target_h) / fade_h

        final_weight = blend_weight * spatial_mask
        final_weight = final_weight[:, :, np.newaxis]

        final_rgb = image_np.astype(np.float32) * (1.0 - final_weight) + decontaminated_rgb * final_weight
        return np.clip(final_rgb, 0, 255).astype(np.uint8)

    def auto_detect_strategy_with_yolo(self, image_rgb, yolo_model_path, target_device="cpu"):
        """
        Runs YOLO to detect organic living targets (people, pets) vs inanimate objects.
        Returns the category classification without altering physical coordinates.
        """
        device_name = f"GPU({target_device.upper()})" if target_device in ['cuda', 'mps'] else "CPU"
        self.log_message.emit(f"Launching YOLO engine to classify subject category | Execution Device: {device_name}...")

        if not os.path.exists(yolo_model_path):
            return 'product'

        if target_device == "cpu":
            _optimize_cpu_core_affinity()

        if PredictWorker._cached_yolo_model is None:
            PredictWorker._cached_yolo_model = YOLO(yolo_model_path)

        PredictWorker._cached_yolo_model.to(target_device)
        model = PredictWorker._cached_yolo_model

        results = model(image_rgb, verbose=False, device=target_device)
        result = results[0]

        strategy = 'product'

        if result.boxes is not None and len(result.boxes) > 0:
            detected_classes = result.boxes.cls.cpu().numpy().astype(int)
            # COCO indices: 0 represents humans, 14-23 represents animals
            is_human_pet = any(c == 0 or (14 <= c <= 23) for c in detected_classes)
            if is_human_pet:
                strategy = 'human_pet'
                self.log_message.emit("📌 YOLO detected [living subject] - routing to MEMatte fine hair engine")
            else:
                self.log_message.emit("📌 YOLO detected [rigid product] - routing to BiRefNet hard-edge engine")
        else:
            self.log_message.emit("⚠️ YOLO detected no subjects. Defaulting to [rigid product] mode.")

        return strategy

    def get_birefnet_initial_mask(self, image_rgb, local_model_path, target_device="cpu"):
        """Extracts initial masks on standard 1024x1024 resolution limits."""
        device_name = f"GPU({target_device.upper()})" if target_device in ['cuda', 'mps'] else "CPU"
        self.log_message.emit(f"Launching local BiRefNet engine (Fixed 1024x1024 resolution) | Execution Device: {device_name}...")

        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local BiRefNet checkpoints missing: {local_model_path}")

        if target_device == "cpu":
            _optimize_cpu_core_affinity()

        if PredictWorker._cached_birefnet_model is None:
            model = AutoModelForImageSegmentation.from_pretrained(
                local_model_path, trust_remote_code=True, local_files_only=True, use_safetensors=True
            )
            if target_device == "cpu":
                model = model.float()
                model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            model.to(target_device).eval()
            PredictWorker._cached_birefnet_model = model
        else:
            model = PredictWorker._cached_birefnet_model

        model_dtype = next(model.parameters()).dtype
        H, W = image_rgb.shape[:2]

        image_pil = PILImage.fromarray(image_rgb)

        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image_pil).unsqueeze(0).to(target_device, dtype=model_dtype)

        with torch.inference_mode():
            if target_device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    preds = model(input_tensor)[-1].sigmoid().cpu().float()
            elif target_device == "cpu":
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                    preds = model(input_tensor)[-1].sigmoid().cpu().float()
            else:
                preds = model(input_tensor)[-1].sigmoid().cpu().float()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize((W, H), resample=PILImage.BILINEAR)

        return np.array(pred_pil)

    def create_trimap_from_alpha(self, alpha, erode_ratio, dilate_ratio):
        """Generates dynamic three-state maps (trimap) based on boundary distances."""
        h, w = alpha.shape
        max_dim = max(h, w)
        fg_erode_size = max(1, int(max_dim * erode_ratio))
        bg_dilate_size = max(1, int(max_dim * dilate_ratio))

        _, fg_mask = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)
        _, bg_mask_inv = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

        close_kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel)
        fg = cv2.erode(fg_mask, np.ones((fg_erode_size, fg_erode_size), np.uint8), iterations=1)
        dilated = cv2.dilate(bg_mask_inv, np.ones((bg_dilate_size, bg_dilate_size), np.uint8), iterations=1)

        trimap = np.full(alpha.shape, 128, dtype=np.uint8)
        trimap[fg == 255] = 255
        trimap[dilated == 0] = 0
        return trimap

    @Slot(object, tuple, object, bool, object)
    def run_predict(self, predictor, point_data, box_data, cumulative, image_rgb=None):
        """Enqueues predictor execution on internal command threads."""
        self._internal_start_signal.emit(predictor, point_data, box_data, cumulative, image_rgb)

    @Slot(object, tuple, object, bool, object)
    def _do_run_predict(self, predictor, point_data, box_data, cumulative, image_rgb):
        """Core prediction pipeline logic."""
        target_device = PredictWorker._get_true_target_device()
        device_name = f"GPU({target_device.upper()})" if target_device in ['cuda', 'mps'] else "CPU"
        self.log_message.emit(f"Entering isolated background thread. Starting image prediction | Execution Device: {device_name}")

        result_mask, success, error_msg = None, False, ""
        start_time = time.time()
        input_point_coords, input_point_labels = point_data
        is_cpu_mode = (target_device == 'cpu')

        try:
            if isinstance(box_data, str) and box_data == "AUTO" and image_rgb is not None:
                app_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.getcwd()
                YOLO_MODEL_PATH = os.path.join(app_path, "checkpoints", "yolo26s.pt")
                LOCAL_BIREFNET_PATH = os.path.join(app_path, "checkpoints", "BiRefNet_local")
                CONFIG_DIR = os.path.join(app_path, "MEMatte", "configs", "MEMatte_B_topk0.25_win_global_long.py")
                CHECKPOINT_DIR = os.path.join(app_path, "checkpoints", "MEMatte_ViTB_DIM.pth")

                STRATEGY_CONFIGS = {
                    'human_pet': {'erode_ratio': 0.10, 'dilate_ratio': 0.15, 'use_decon': True, 'clip_low': 20,
                                  'clip_high': 240},
                    'product': {'erode_ratio': 0.005, 'dilate_ratio': 0.03, 'use_decon': False, 'clip_low': 10,
                                'clip_high': 245},
                    'logo': {'erode_ratio': 0.005, 'dilate_ratio': 0.01, 'use_decon': False, 'clip_low': 60,
                             'clip_high': 190}
                }

                if target_device == "cuda":
                    torch.cuda.empty_cache()

                TARGET_STRATEGY = self.auto_detect_strategy_with_yolo(image_rgb, YOLO_MODEL_PATH, target_device)
                current_config = STRATEGY_CONFIGS.get(TARGET_STRATEGY, STRATEGY_CONFIGS['product'])
                initial_alpha = self.get_birefnet_initial_mask(image_rgb, LOCAL_BIREFNET_PATH, target_device)

                if TARGET_STRATEGY == 'human_pet':
                    self.log_message.emit(f"🔥 Launching MEMatte fine hair engine (Adaptive ROI local refinement) | Execution Device: {device_name}...")
                    H, W = image_rgb.shape[:2]
                    alpha_bin = (initial_alpha > 128).astype(np.uint8)
                    y_indices, x_indices = np.where(alpha_bin > 0)

                    pad_ratio = 0.20 if not is_cpu_mode else 0.15
                    MAX_ROI_DIM = 1536.0 if not is_cpu_mode else 768.0

                    if len(y_indices) > 0:
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        pad_x, pad_y = max(10, int((x_max - x_min) * pad_ratio)), max(10, int((y_max - y_min) * pad_ratio))
                        x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
                        x2, y2 = min(W, x_max + pad_x), min(H, y_max + pad_y)
                    else:
                        x1, y1, x2, y2 = 0, 0, W, H

                    roi_rgb = image_rgb[y1:y2, x1:x2]
                    roi_initial_alpha = initial_alpha[y1:y2, x1:x2]
                    roi_h, roi_w = roi_rgb.shape[:2]
                    roi_trimap = self.create_trimap_from_alpha(roi_initial_alpha, current_config['erode_ratio'],
                                                               current_config['dilate_ratio'])

                    scale = min(1.0, MAX_ROI_DIM / max(roi_w, roi_h))
                    infer_w = max(32, (int(roi_w * scale) // 32) * 32)
                    infer_h = max(32, (int(roi_h * scale) // 32) * 32)

                    if scale < 1.0 or infer_w != roi_w or infer_h != roi_h:
                        infer_rgb = cv2.resize(roi_rgb, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                        infer_trimap = cv2.resize(roi_trimap, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        infer_rgb, infer_trimap = roi_rgb, roi_trimap

                    if PredictWorker._cached_mematte_model is None:
                        cfg = LazyConfig.load(CONFIG_DIR)
                        cfg.model.teacher_backbone = None
                        cfg.model.backbone.max_number_token = 18500
                        model = instantiate(cfg.model)
                        if is_cpu_mode:
                            model = model.float()
                        DetectionCheckpointer(model).load(CHECKPOINT_DIR)
                        if is_cpu_mode:
                            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                        model.to(target_device).eval()
                        PredictWorker._cached_mematte_model = model
                    else:
                        model = PredictWorker._cached_mematte_model

                    img_tensor = F.to_tensor(PILImage.fromarray(infer_rgb)).unsqueeze(0).to(target_device)
                    tri_tensor = F.to_tensor(PILImage.fromarray(infer_trimap).convert('L')).unsqueeze(0).to(target_device)

                    with torch.inference_mode():
                        if target_device == "cuda":
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                output = model({'image': img_tensor, 'trimap': tri_tensor}, patch_decoder=True)[0]
                        else:
                            output = model({'image': img_tensor, 'trimap': tri_tensor}, patch_decoder=True)[0]

                        output = output['phas'].flatten(0, 2).float()
                        trimap_mask = tri_tensor.squeeze(0).squeeze(0)
                        output[trimap_mask == 0], output[trimap_mask == 1] = 0, 1

                        alpha_infer = np.array(F.to_pil_image(output.cpu()))

                    if scale < 1.0 or infer_w != roi_w or infer_h != roi_h:
                        alpha_roi = cv2.resize(alpha_infer, (roi_w, roi_h), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        alpha_roi = alpha_infer

                    alpha_roi = self.refine_alpha_natural_clarity(
                        self.fill_alpha_internal_holes(alpha_roi, max_hole_area=3000), current_config['clip_low'],
                        current_config['clip_high']
                    )

                    full_alpha = initial_alpha.copy()

                    edge_blend = np.ones((roi_h, roi_w), dtype=np.float32)
                    fade_px = min(40, roi_w // 2, roi_h // 2)
                    if fade_px > 0:
                        for i in range(fade_px):
                            alpha_fade = (i / float(fade_px)) ** 2
                            edge_blend[:, i] = np.minimum(edge_blend[:, i], alpha_fade)
                            edge_blend[:, -1 - i] = np.minimum(edge_blend[:, -1 - i], alpha_fade)
                            edge_blend[i, :] = np.minimum(edge_blend[i, :], alpha_fade)
                            edge_blend[-1 - i, :] = np.minimum(edge_blend[-1 - i, :], alpha_fade)

                    orig_roi = full_alpha[y1:y2, x1:x2].astype(np.float32)
                    new_roi = alpha_roi.astype(np.float32)
                    blended_roi = orig_roi * (1.0 - edge_blend) + new_roi * edge_blend
                    full_alpha[y1:y2, x1:x2] = blended_roi.astype(np.uint8)

                    full_rgb = image_rgb.copy()
                    if current_config['use_decon']:
                        full_rgb = self.natural_hair_color_decontamination(full_rgb, full_alpha)

                    alpha_final_res = np.array(full_alpha.astype(np.float32) / 255.0, copy=True, order='C')
                    rgb_final_res = np.array(full_rgb, copy=True, order='C')
                    result_mask = (alpha_final_res, rgb_final_res)
                    success = True

                else:
                    alpha_np = self.refine_alpha_natural_clarity(
                        self.fill_alpha_internal_holes(initial_alpha, max_hole_area=3000), current_config['clip_low'],
                        current_config['clip_high']
                    )
                    result_mask = (
                        np.array(alpha_np.astype(np.float32) / 255.0, copy=True, order='C'),
                        np.array(image_rgb, copy=True, order='C')
                    )
                    success = True

            else:
                if predictor is None:
                    raise ValueError("Target image predictor is not initialized.")
                use_multimask = (input_point_coords is not None) and (box_data is None)
                mask_tensor = image_rgb if isinstance(image_rgb, torch.Tensor) else None

                masks, scores, _ = predictor.predict(
                    point_coords=input_point_coords, point_labels=input_point_labels, box=box_data,
                    mask_input=mask_tensor, multimask_output=use_multimask
                )

                if masks is not None and len(masks) > 0:
                    mask_output = masks[np.argmax(scores)] if use_multimask else masks[0]
                    result_mask = mask_output.cpu().numpy() if hasattr(mask_output, "cpu") else mask_output
                    if result_mask.ndim == 3:
                        result_mask = result_mask.squeeze(0)
                    result_mask = np.array(result_mask > 0, copy=True, order='C')
                    success = True

        except Exception as e:
            error_msg = str(e)
            traceback.print_exc()

        finally:
            self.log_message.emit(f"Background prediction complete. Elapsed: {time.time() - start_time:.2f}s | Target Device: {device_name}")
            self.finished.emit(result_mask, success, error_msg)


class BatchMattingWorker(PredictWorker):
    """
    Worker tasked with processing batch automatic segmentation actions.
    Provides UI progress callback loops.
    """
    single_result_ready = Signal(int, str, object, object, str)

    @Slot(list, str)
    def run_batch(self, file_paths, output_dir):
        target_device = self._get_true_target_device()
        device_name = f"GPU({target_device.upper()})" if target_device in ['cuda', 'mps'] else "CPU"
        self.log_message.emit(f"🚀 Starting batch auto-matting for {len(file_paths)} items | Execution Device: {device_name}")

        success_count = 0
        total = len(file_paths)

        app_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.getcwd()
        YOLO_MODEL_PATH = os.path.join(app_path, "checkpoints", "yolo26s.pt")
        LOCAL_BIREFNET_PATH = os.path.join(app_path, "checkpoints", "BiRefNet_local")
        CONFIG_DIR = os.path.join(app_path, "MEMatte", "configs", "MEMatte_B_topk0.25_win_global_long.py")
        CHECKPOINT_DIR = os.path.join(app_path, "checkpoints", "MEMatte_ViTB_DIM.pth")

        STRATEGY_CONFIGS = {
            'human_pet': {'erode_ratio': 0.10, 'dilate_ratio': 0.15, 'use_decon': True, 'clip_low': 20,
                          'clip_high': 240},
            'product': {'erode_ratio': 0.005, 'dilate_ratio': 0.03, 'use_decon': False, 'clip_low': 10,
                        'clip_high': 245},
            'logo': {'erode_ratio': 0.005, 'dilate_ratio': 0.01, 'use_decon': False, 'clip_low': 60, 'clip_high': 190}
        }

        try:
            for i, file_path in enumerate(file_paths):
                if self._is_cancelled:
                    break

                self.progress.emit(int((i / total) * 100), f"Processing: {os.path.basename(file_path)}")
                bgra_img = None
                mask_bool = None
                err_msg = ""

                try:
                    cv_img = imread_unicode(file_path, cv2.IMREAD_COLOR)
                    if cv_img is None:
                        raise ValueError("Failed to read image data.")

                    image_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    H, W = image_rgb.shape[:2]

                    if target_device == "cuda":
                        torch.cuda.empty_cache()

                    # 1. Subject Classification & Coarse Mask Extraction
                    TARGET_STRATEGY = self.auto_detect_strategy_with_yolo(image_rgb, YOLO_MODEL_PATH, target_device)
                    current_config = STRATEGY_CONFIGS.get(TARGET_STRATEGY, STRATEGY_CONFIGS['product'])
                    initial_alpha = self.get_birefnet_initial_mask(image_rgb, LOCAL_BIREFNET_PATH, target_device)

                    # 2. Strategy-based Matting Refinement
                    if TARGET_STRATEGY == 'human_pet':
                        alpha_bin = (initial_alpha > 128).astype(np.uint8)
                        y_indices, x_indices = np.where(alpha_bin > 0)

                        is_cpu_mode = (target_device == "cpu")
                        pad_ratio = 0.15 if is_cpu_mode else 0.20
                        MAX_ROI_DIM = 768.0 if is_cpu_mode else 1536.0

                        if len(y_indices) > 0:
                            x_min, x_max = np.min(x_indices), np.max(x_indices)
                            y_min, y_max = np.min(y_indices), np.max(y_indices)
                            pad_x, pad_y = max(10, int((x_max - x_min) * pad_ratio)), max(10, int((y_max - y_min) * pad_ratio))
                            x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
                            x2, y2 = min(W, x_max + pad_x), min(H, y_max + pad_y)
                        else:
                            x1, y1, x2, y2 = 0, 0, W, H

                        roi_rgb = image_rgb[y1:y2, x1:x2]
                        roi_trimap = self.create_trimap_from_alpha(initial_alpha[y1:y2, x1:x2],
                                                                   current_config['erode_ratio'],
                                                                   current_config['dilate_ratio'])

                        scale = min(1.0, MAX_ROI_DIM / max(roi_rgb.shape[1], roi_rgb.shape[0]))
                        infer_w, infer_h = max(32, (int(roi_rgb.shape[1] * scale) // 32) * 32), max(32, (int(roi_rgb.shape[0] * scale) // 32) * 32)

                        infer_rgb = cv2.resize(roi_rgb, (infer_w, infer_h)) if scale < 1.0 else roi_rgb
                        infer_trimap = cv2.resize(roi_trimap, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else roi_trimap

                        if PredictWorker._cached_mematte_model is None:
                            cfg = LazyConfig.load(CONFIG_DIR)
                            cfg.model.teacher_backbone = None
                            cfg.model.backbone.max_number_token = 18500
                            model = instantiate(cfg.model)
                            if is_cpu_mode:
                                model = model.float()
                            DetectionCheckpointer(model).load(CHECKPOINT_DIR)
                            if is_cpu_mode:
                                model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                            model.to(target_device).eval()
                            PredictWorker._cached_mematte_model = model
                        else:
                            model = PredictWorker._cached_mematte_model

                        img_tensor = F.to_tensor(PILImage.fromarray(infer_rgb)).unsqueeze(0).to(target_device)
                        tri_tensor = F.to_tensor(PILImage.fromarray(infer_trimap).convert('L')).unsqueeze(0).to(target_device)

                        with torch.inference_mode():
                            if target_device == "cuda":
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    output = model({'image': img_tensor, 'trimap': tri_tensor}, patch_decoder=True)[0]
                            else:
                                output = model({'image': img_tensor, 'trimap': tri_tensor}, patch_decoder=True)[0]

                            output = output['phas'].flatten(0, 2).float()
                            trimap_mask = tri_tensor.squeeze(0).squeeze(0)
                            output[trimap_mask == 0], output[trimap_mask == 1] = 0, 1
                            alpha_infer = np.array(F.to_pil_image(output))

                        alpha_roi = cv2.resize(alpha_infer, (roi_rgb.shape[1], roi_rgb.shape[0])) if scale < 1.0 else alpha_infer
                        alpha_roi = self.refine_alpha_natural_clarity(self.fill_alpha_internal_holes(alpha_roi),
                                                                      current_config['clip_low'],
                                                                      current_config['clip_high'])

                        full_alpha = initial_alpha.copy()

                        edge_blend = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
                        fade_px = min(40, (x2 - x1) // 2, (y2 - y1) // 2)
                        if fade_px > 0:
                            for i_f in range(fade_px):
                                alpha_fade = (i_f / float(fade_px)) ** 2
                                edge_blend[:, i_f] = np.minimum(edge_blend[:, i_f], alpha_fade)
                                edge_blend[:, -1 - i_f] = np.minimum(edge_blend[:, -1 - i_f], alpha_fade)
                                edge_blend[i_f, :] = np.minimum(edge_blend[i_f, :], alpha_fade)
                                edge_blend[-1 - i_f] = np.minimum(edge_blend[-1 - i_f], alpha_fade)

                        orig_roi = full_alpha[y1:y2, x1:x2].astype(np.float32)
                        new_roi = alpha_roi.astype(np.float32)
                        blended_roi = orig_roi * (1.0 - edge_blend) + new_roi * edge_blend
                        full_alpha[y1:y2, x1:x2] = blended_roi.astype(np.uint8)
                        final_alpha_u8 = full_alpha

                        if current_config['use_decon']:
                            full_rgb = image_rgb.copy()
                            full_rgb = self.natural_hair_color_decontamination(full_rgb, final_alpha_u8)
                            cv_img = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)

                    else:
                        final_alpha_u8 = self.refine_alpha_natural_clarity(
                            self.fill_alpha_internal_holes(initial_alpha), current_config['clip_low'],
                            current_config['clip_high']
                        )

                    bgra_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
                    bgra_img[:, :, 3] = final_alpha_u8
                    mask_bool = final_alpha_u8 > 128

                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    traceback.print_exc()

                self.single_result_ready.emit(i, file_path, bgra_img, mask_bool, err_msg)

            self.progress.emit(100, f"Batch processing complete! Successful: {success_count}/{total}")
            self.finished.emit(success_count, True, "")

        except Exception as e:
            self.finished.emit(success_count, False, str(e))


class FrameExtractorWorker(WorkerBase):
    """
    Video stream frame and thumbnail extractor utilizing the PyAV backend.
    """
    @Slot(str, str, int)
    def run_extract(self, video_path, temp_frame_dir, every_n):
        self.log_message.emit(f"Worker thread started: [PyAV] Parsing and extracting video frames from {os.path.basename(video_path)}...")

        frame_count = 0
        fps = VIDEO_DEFAULT_FPS
        width, height = 0, 0
        is_gif = video_path.lower().endswith('.gif')
        gif_duration_ms = int(1000 / VIDEO_DEFAULT_FPS)
        success = False
        error_msg = ""
        start_time = time.time()
        thumbnail_paths = []

        try:
            os.makedirs(temp_frame_dir, exist_ok=True)
            thumb_dir = os.path.join(temp_frame_dir, "thumbnails")
            os.makedirs(thumb_dir, exist_ok=True)

            container = av.open(video_path)
            stream = container.streams.video[0]
            width = stream.width
            height = stream.height

            fps_val = stream.average_rate
            if not fps_val:
                fps_val = stream.r_frame_rate
            if fps_val:
                fps = float(fps_val)
            else:
                fps = VIDEO_DEFAULT_FPS

            total_vid_frames = stream.frames
            if not total_vid_frames or total_vid_frames <= 0:
                if stream.duration and stream.time_base:
                    total_vid_frames = int(float(stream.duration * stream.time_base) * fps)
                else:
                    total_vid_frames = 1

            if is_gif:
                gif_duration_ms = int(1000 / fps) if fps > 0 else int(1000 / VIDEO_DEFAULT_FPS)

            total_to_extract = (total_vid_frames + every_n - 1) // every_n if every_n > 0 else total_vid_frames

            thumb_h = 50
            aspect = width / height if height > 0 else 16 / 9
            thumb_w = max(30, int(thumb_h * aspect))

            local_idx = 0
            extracted_count = 0

            for frame in container.decode(video=0):
                if self._is_cancelled:
                    container.close()
                    raise InterruptedError("Frame extraction cancelled by user.")

                if every_n > 1 and (local_idx % every_n != 0):
                    local_idx += 1
                    continue

                bgr_array = frame.to_ndarray(format='bgr24')
                frame_output_path = os.path.join(temp_frame_dir, f"{extracted_count:05d}.jpg")

                is_success, im_buf_arr = cv2.imencode('.jpg', bgr_array, [cv2.IMWRITE_JPEG_QUALITY, 100])
                if is_success:
                    with open(frame_output_path, "wb") as f:
                        f.write(im_buf_arr)
                else:
                    cv2.imwrite(frame_output_path, bgr_array, [cv2.IMWRITE_JPEG_QUALITY, 100])

                thumb_cv = cv2.resize(bgr_array, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                thumb_path = os.path.join(thumb_dir, f"{extracted_count}.jpg")
                cv2.imwrite(thumb_path, thumb_cv, [cv2.IMWRITE_JPEG_QUALITY, 75])
                thumbnail_paths.append(thumb_path)

                extracted_count += 1
                local_idx += 1

                progress_pct = min(99, int((extracted_count / max(1, total_to_extract)) * 100))
                self.progress.emit(progress_pct, f"PyAV Frame Decoding: {extracted_count}/{total_to_extract}")

            container.close()
            frame_count = extracted_count
            success = True
            self.log_message.emit(f"[PyAV] Processing complete: Extracted {frame_count} frames in {time.time() - start_time:.2f}s")
            self.progress.emit(100, "Frame extraction complete.")

        except InterruptedError:
            error_msg = "Frame extraction cancelled by user."
            success = False
            self.progress.emit(0, "Operation cancelled.")
        except Exception as e:
            error_msg = f"Error extracting frames: {e}"
            traceback.print_exc()
            success = False
            frame_count = 0
            self.progress.emit(0, "Frame extraction failed.")
        finally:
            result_data = (
                frame_count, fps, width, height, is_gif,
                gif_duration_ms if is_gif else int(1000 / fps if fps > 0 else 40),
                thumbnail_paths
            ) if success else None
            self.finished.emit(result_data, success, error_msg)


class VideoSegmentationPropagateWorker(WorkerBase):
    """
    Video state evaluation propagation handler. Tracks structural elements across frames sequentially.
    """

    @Slot(object, object, dict, int, int)
    def run_propagation(self, video_predictor, inference_state, valid_targets, start_frame, end_frame):
        import time
        import gc

        real_target_device = PredictWorker._get_true_target_device()
        device_obj = torch.device(real_target_device)

        processed_masks = collections.defaultdict(dict)
        success = False
        error_msg = ""
        total_clip_frames = end_frame - start_frame + 1

        try:
            self.progress.emit(2, "Preparing video feature tracking streams...")
            time.sleep(0.05)

            if inference_state is None:
                raise ValueError("No ready video layer feature state detected. Unable to propagate tracking.")

            self.progress.emit(10, "Initializing layer localization and synchronizing tracking sequence...")
            time.sleep(0.05)

            use_autocast = (device_obj.type == 'cuda')
            autocast_dtype = torch.float16
            autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else nullcontext()

            processed_frames = 0
            last_update_time = time.time()

            with torch.inference_mode(), autocast_ctx:
                for out_local_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                    if self._is_cancelled:
                        raise InterruptedError("Video segmentation cancelled.")

                    global_idx = out_local_idx + start_frame
                    frame_results = {}

                    if out_obj_ids is not None and out_mask_logits is not None:
                        for i, obj_id_tensor in enumerate(out_obj_ids):
                            obj_id = int(obj_id_tensor.item()) if hasattr(obj_id_tensor, 'item') else int(obj_id_tensor)

                            # [修复 Issue 1]: 丢弃当前对象在其标注帧之前的时序追踪预测结果
                            if obj_id in valid_targets:
                                target_ann_frame = valid_targets[obj_id].get('annotation_frame')
                                if target_ann_frame is not None and global_idx < target_ann_frame:
                                    continue

                            mask_bool = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                            if mask_bool.dtype != bool:
                                mask_bool = mask_bool > 0
                            frame_results[obj_id] = np.array(mask_bool, copy=True, order='C')

                    processed_masks[global_idx] = frame_results
                    processed_frames += 1

                    del out_obj_ids, out_mask_logits

                    if device_obj.type == 'cuda':
                        torch.cuda.synchronize()

                    curr_time = time.time()
                    if curr_time - last_update_time >= 0.1 or processed_frames == total_clip_frames:
                        pct = 15 + int((processed_frames / total_clip_frames) * 83)
                        self.progress.emit(pct,
                                           f"Intelligent frame parsing: {processed_frames}/{total_clip_frames} frames")
                        last_update_time = curr_time

                    time.sleep(0.015)

            if not self._is_cancelled:
                success = True
                self.progress.emit(100, "Video tracking propagation complete.")

        except InterruptedError as ie:
            error_msg = str(ie)
            success = False
        except Exception as e:
            error_msg = f"Video segmentation propagation failed: {e}"
            traceback.print_exc()
            success = False
        finally:
            if device_obj.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        self.finished.emit(dict(processed_masks), success, error_msg)


class VideoMatAnyoneWorker(WorkerBase):
    """
    完全对齐官方 MatAnyone2 WebUI 的时序抠图追踪工作线程。
    - 若不勾选发丝 (use_matanyone=False)：运行标准 SAM2 视频时序多目标并行追踪。
    - 若勾选发丝 (use_matanyone=True)：对每个目标独立进行隔离的正向传播与发丝级精雕，
      有效防止多目标在不同帧标注时产生时间戳投影错位和交叉污染。
    """
    # 发送当前正在编码/追踪的物理绝对帧索引，用于主界面实时刷新
    frame_updated = Signal(int)

    @Slot(object, object, object, str, str, dict, int, int, int, int, bool, int)
    def run_sam2_guided_matanyone_propagation(
            self, sam_predictor, inference_state, mat_model, temp_frame_dir: str, clip_sandbox_dir: str,
            valid_targets: dict,
            start_frame: int, end_frame: int, erode_kernel_size: int = 10, dilate_kernel_size: int = 10,
            use_matanyone: bool = False, n_warmup: int = 10
    ):
        import gc
        import torch
        import numpy as np
        import cv2
        import collections
        import traceback
        import time

        self.log_message.emit(
            f"启动抠图追踪流水线。发丝大模型={use_matanyone}, 结束帧={end_frame}"
        )

        success = False
        error_msg = ""
        final_processed_masks = collections.defaultdict(dict)

        real_target_device = 'cpu'
        if torch is not None:
            if torch.cuda.is_available():
                real_target_device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                real_target_device = 'mps'
        device_obj = torch.device(real_target_device)

        def force_clean_memory():
            gc.collect()
            if device_obj.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            elif device_obj.type == 'mps':
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        try:
            force_clean_memory()
            total_clip_frames = end_frame - start_frame + 1
            ordered_tids = sorted(list(valid_targets.keys()))

            # 用于 MatAnyone 时序追踪的获取本地帧路径（使用全局索引与 temp_frame_dir）
            def _get_frame_path(f_idx):
                padded = os.path.join(temp_frame_dir, f"{f_idx:05d}.jpg")
                return padded if os.path.exists(padded) else os.path.join(temp_frame_dir, f"{f_idx}.jpg")

            # =========================================================================
            # 分支 A: 纯标准 SAM2 视频时序追踪 (未勾选发丝，适合硬边刚性主体)
            # =========================================================================
            if not use_matanyone:
                if sam_predictor is None:
                    raise ValueError("SAM2 预测器未加载，无法执行基础追踪。")

                self.progress.emit(5, "正在初始化时序结构描述信息...")

                use_autocast = (device_obj.type == 'cuda')
                autocast_dtype = torch.float16
                autocast_ctx = torch.autocast(device_type="cuda",
                                              dtype=autocast_dtype) if use_autocast else nullcontext()

                with torch.inference_mode(), autocast_ctx:
                    sam_inference_state = sam_predictor.init_state(
                        video_path=clip_sandbox_dir,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                        async_loading_frames=True
                    )

                    self.progress.emit(10, "启动特征编码器主动预热...")
                    last_emit_time = 0.0

                    batch_size = 1
                    if isinstance(sam_inference_state, dict) and "batch_size" in sam_inference_state:
                        batch_size = sam_inference_state["batch_size"]

                    for c in range(total_clip_frames):
                        if self._is_cancelled:
                            raise InterruptedError("操作已被用户取消。")

                        try:
                            if hasattr(sam_predictor, "_get_image_feature"):
                                try:
                                    sam_predictor._get_image_feature(sam_inference_state, c, batch_size)
                                except TypeError:
                                    sam_predictor._get_image_feature(sam_inference_state, c)
                            elif hasattr(sam_predictor, "_prepare_backbone_features"):
                                try:
                                    sam_predictor._prepare_backbone_features(sam_inference_state, c, batch_size)
                                except TypeError:
                                    sam_predictor._prepare_backbone_features(sam_inference_state, c)
                            elif hasattr(sam_predictor, "get_image_features"):
                                try:
                                    sam_predictor.get_image_features(sam_inference_state, c, batch_size)
                                except TypeError:
                                    sam_predictor.get_image_features(sam_inference_state, c)
                        except Exception as e:
                            self.log_message.emit(f"警告：预热第 {c} 帧特征时发生非致命错误: {e}")

                        curr_time = time.time()
                        if curr_time - last_emit_time >= 0.033 or c == total_clip_frames - 1:
                            current_abs_frame = start_frame + c
                            self.frame_updated.emit(current_abs_frame)
                            last_emit_time = curr_time

                        pct = 10 + int((c / max(1, total_clip_frames)) * 30)
                        self.progress.emit(pct, f"时序特征预编码中: {c + 1}/{total_clip_frames} 帧")
                        time.sleep(0.002)

                        if c % 25 == 0:
                            force_clean_memory()

                self.progress.emit(45, "正在向 SAM2 时序引擎注册交互标注...")
                for tid, data in valid_targets.items():
                    frame_idx = data.get('annotation_frame')
                    if frame_idx is not None:
                        local_ann_frame = frame_idx - start_frame
                        input_points_np, input_labels_np, box_np = None, None, None
                        has_points = bool(data.get('points'))
                        has_box = bool(data.get('box') is not None)

                        if has_points:
                            pts = data['points']
                            input_points_np = np.array([(p[0], p[1]) for p in pts], dtype=np.float32)
                            if input_points_np.ndim == 1 and input_points_np.size > 0:
                                input_points_np = input_points_np[np.newaxis, :]
                            input_labels_np = np.array([p[2] for p in pts], dtype=np.int32)
                        if has_box:
                            box_np = np.array(data['box'], dtype=np.float32)

                        if has_points or has_box:
                            sam_predictor.add_new_points_or_box(
                                inference_state=sam_inference_state,
                                frame_idx=local_ann_frame, obj_id=tid,
                                points=input_points_np, labels=input_labels_np, box=box_np,
                                clear_old_points=True
                            )

                self.progress.emit(55, "特征预热已完成，标准 SAM2 时序追踪传播开始...")
                processed_frames = 0
                last_propagate_emit_time = 0.0

                with torch.inference_mode(), autocast_ctx:
                    for out_local_idx, out_obj_ids, out_mask_logits in sam_predictor.propagate_in_video(
                            sam_inference_state):
                        if self._is_cancelled:
                            raise InterruptedError("操作已被用户取消。")

                        global_idx = out_local_idx + start_frame
                        frame_results = {}

                        if out_obj_ids is not None and out_mask_logits is not None:
                            for i_obj, obj_id_tensor in enumerate(out_obj_ids):
                                obj_id = int(obj_id_tensor.item()) if hasattr(obj_id_tensor, 'item') else int(
                                    obj_id_tensor)
                                if obj_id in ordered_tids:
                                    target_ann_frame = valid_targets[obj_id].get('annotation_frame')
                                    if target_ann_frame is not None and global_idx < target_ann_frame:
                                        continue

                                    mask_bool = (out_mask_logits[i_obj] > 0.0).cpu().numpy().squeeze()
                                    if mask_bool.dtype != bool:
                                        mask_bool = mask_bool > 0
                                    frame_results[obj_id] = np.array(mask_bool, copy=True, order='C')

                        final_processed_masks[global_idx] = frame_results
                        processed_frames += 1

                        curr_time = time.time()
                        if curr_time - last_propagate_emit_time >= 0.033 or processed_frames == total_clip_frames:
                            self.frame_updated.emit(global_idx)
                            last_propagate_emit_time = curr_time

                        pct = 55 + int((processed_frames / total_clip_frames) * 43)
                        self.progress.emit(pct, f"SAM2 分割计算中: {processed_frames}/{total_clip_frames} 帧")

                        if processed_frames % 20 == 0:
                            force_clean_memory()

                if sam_inference_state is not None:
                    try:
                        sam_predictor.reset_state(sam_inference_state)
                    except Exception:
                        pass

            # =========================================================================
            # 分支 B: 纯 MatAnyone2 全局视频发丝抠图（已支持多目标隔离传播）
            # =========================================================================
            else:
                if mat_model is None:
                    raise ValueError("MatAnyone2 发丝精调大模型未加载，请等待模型加载完成。")

                from core.models.matanyone2.inference.inference_core import InferenceCore
                from core.models.matanyone2.matanyone2_wrapper import matanyone2

                for tid_idx, tid in enumerate(ordered_tids):
                    if self._is_cancelled:
                        raise InterruptedError("操作已被用户取消。")

                    data = valid_targets[tid]
                    ann_frame_global = data.get('annotation_frame')
                    if ann_frame_global is None:
                        continue

                    obj_mask = data.get('initial_mask')
                    if obj_mask is None:
                        continue

                    self.progress.emit(
                        10 + int((tid_idx / len(ordered_tids)) * 10),
                        f"正在为对象 {tid + 1} 提取后续视频帧(following_frames)..."
                    )

                    ann_frame_path = _get_frame_path(ann_frame_global)
                    ann_frame_bgr = cv2.imread(ann_frame_path)
                    if ann_frame_bgr is None:
                        continue
                    H, W = ann_frame_bgr.shape[:2]
                    del ann_frame_bgr

                    following_frames = []
                    for f_idx in range(ann_frame_global, end_frame + 1):
                        f_path = _get_frame_path(f_idx)
                        f_bgr = cv2.imread(f_path)
                        if f_bgr is None:
                            f_bgr = np.zeros((H, W, 3), dtype=np.uint8)
                        f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
                        following_frames.append(f_rgb)

                    template_mask_scaled = np.zeros((H, W), dtype=np.int32)
                    mask_bool = (obj_mask > 0.5) if np.issubdtype(obj_mask.dtype, np.floating) else obj_mask
                    template_mask_scaled[mask_bool] = 255

                    matanyone_processor = InferenceCore(mat_model, cfg=mat_model.cfg)
                    matanyone_processor.device = device_obj

                    base_progress = 20 + int((tid_idx / len(ordered_tids)) * 70)
                    progress_range = int(70 / len(ordered_tids))

                    def make_fw_cb(t_id, base_p, p_range):
                        last_emit = [0.0]
                        display_id = t_id + 1

                        def _fw_cb(curr, tot):
                            if self._is_cancelled:
                                raise InterruptedError("操作已被用户取消。")

                            pct = base_p + int((curr / max(1, tot)) * p_range)
                            self.progress.emit(pct, f"对象 {display_id} 发丝时序追踪渲染: {curr}/{tot} 帧")

                            curr_time = time.time()
                            if curr_time - last_emit[0] >= 0.033 or curr == tot:
                                # =========================================================================
                                # 【核心修复】：防止 curr=0 或 1-based 转换计算产生的下限越界闪烁
                                # =========================================================================
                                current_abs_frame = ann_frame_global + max(0, curr - 1)
                                current_abs_frame = min(current_abs_frame, end_frame)  # 严格限制上限
                                self.frame_updated.emit(current_abs_frame)
                                last_emit[0] = curr_time

                        return _fw_cb

                    with torch.inference_mode():
                        with torch.autocast(device_type=device_obj.type, enabled=(device_obj.type == 'cuda')):
                            _, alphas = matanyone2(
                                matanyone_processor,
                                following_frames,
                                template_mask_scaled,
                                r_erode=erode_kernel_size,
                                r_dilate=dilate_kernel_size,
                                n_warmup=n_warmup,
                                progress_callback=make_fw_cb(tid, base_progress, progress_range)
                            )

                    if hasattr(matanyone_processor, 'clear_memory'):
                        matanyone_processor.clear_memory()
                    force_clean_memory()

                    for i, alpha_res in enumerate(alphas):
                        global_idx = ann_frame_global + i
                        if hasattr(alpha_res, 'detach'):
                            target_alpha = alpha_res.detach().cpu().numpy().squeeze()
                        else:
                            target_alpha = np.array(alpha_res, copy=True).squeeze()

                        if target_alpha.dtype == np.uint8:
                            target_alpha = target_alpha.astype(np.float32) / 255.0
                        target_alpha = np.clip(target_alpha, 0.0, 1.0)

                        final_processed_masks[global_idx][tid] = target_alpha

                    del following_frames, alphas
                    force_clean_memory()

            success = True
            self.progress.emit(100, "视频抠图追踪与渲染完成！")

        except InterruptedError as ie:
            error_msg = str(ie)
            success = False
        except Exception as e:
            error_msg = f"视频时序传播失败: {e}"
            traceback.print_exc()
            success = False
        finally:
            force_clean_memory()
            self.finished.emit(dict(final_processed_masks), success, error_msg)



class SaveWorker(QObject):
    """
    高保真、高性能保存工作线程类
    核心特性：音视频双阶段解耦导出，彻底解决多轨音频混合时的噪点、卡顿与处理极慢的性能瓶颈。
    """
    progress = Signal(int, str)  # 进度信号 (百分比, 提示文本)
    finished = Signal(bool, str, str)  # 完成信号 (是否成功, 保存路径, 错误信息)
    log_message = Signal(str)  # 日志调试信号
    error = Signal(str)  # 异常错误信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_cancelled = False

    @Slot(str, object, str, dict)
    def run_save(self, data_type: str, save_data: object, save_path: str, options: Optional[dict] = None):
        """
        统一保存任务执行入口
        """
        self.is_cancelled = False
        self.log_message.emit(f"开始执行保存任务，类型: {data_type}，路径: {save_path}")

        try:
            if data_type == "enhanced_image":
                self._save_enhanced_image(save_data, save_path)
            elif data_type == "segmented_image":
                self._save_segmented_image(save_data, save_path)
            elif data_type == "segmented_video":
                self._save_segmented_video(save_data, save_path, options)
            else:
                err_msg = f"未知的保存任务类型: {data_type}"
                self.error.emit(err_msg)
                self.finished.emit(False, "", err_msg)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(False, "", str(e))

    def cancel(self):
        """
        用户中止导出任务
        """
        self.is_cancelled = True
        self.log_message.emit("收到用户取消导出请求。")

    def _save_enhanced_image(self, enhanced_image_cv: np.ndarray, save_path: str):
        """
        保存高清增强图像
        """
        try:
            self.progress.emit(30, "正在解码超高清图像像素...")

            if len(enhanced_image_cv.shape) == 3 and enhanced_image_cv.shape[2] == 4:
                ext = os.path.splitext(save_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.jfif']:
                    bgr = cv2.cvtColor(enhanced_image_cv, cv2.COLOR_BGRA2BGR)
                    enhanced_image_cv = bgr

            self.progress.emit(70, "正在压缩并安全持久化到磁盘...")
            success = cv2.imwrite(save_path, enhanced_image_cv, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            if success:
                self.progress.emit(100, "图像保存成功。")
                self.finished.emit(True, save_path, "")
            else:
                self.finished.emit(False, "", "由于操作系统磁盘写入返回 False，保存失败")
        except Exception as e:
            self.finished.emit(False, "", f"图像保存发生错误: {e}")

    def _save_segmented_image(self, save_data: tuple, save_path: str):
        """
        异步保存抠图图像：在子线程完成 4K/8K 级边界精雕和自动边界裁剪
        """
        try:
            original_cv_image_full_res, working_res_mask, scale_factor, refine_params = save_data
            h_orig, w_orig = original_cv_image_full_res.shape[:2]

            self.progress.emit(15, "正在生成超高清蒙版细部边界...")

            if working_res_mask.shape[:2] != (h_orig, w_orig):
                final_mask = upscale_mask_with_guidance(
                    low_res_mask=working_res_mask,
                    high_res_guide=original_cv_image_full_res,
                    subject_type="product"
                )
            else:
                final_mask = working_res_mask.astype(np.float32)

            if self.is_cancelled:
                self.finished.emit(False, "", "用户取消了导出")
                return

            self.progress.emit(45, "正在运行双引擎发丝边界优化...")
            final_mask_refined = _static_apply_mask_refinements(
                final_mask, original_cv_image_full_res, refine_params
            )

            if self.is_cancelled:
                self.finished.emit(False, "", "用户取消了导出")
                return

            self.progress.emit(75, "正在编码并裁剪输出透明 PNG 通道...")

            mask_u8 = (final_mask_refined > 0.1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                x, y, w, h = cv2.boundingRect(np.concatenate(contours))
                cropped_image = original_cv_image_full_res[y:y + h, x:x + w]
                cropped_mask = final_mask_refined[y:y + h, x:x + w]
            else:
                cropped_image = original_cv_image_full_res
                cropped_mask = final_mask_refined

            alpha_channel = (cropped_mask * 255.0).clip(0, 255).astype(np.uint8)
            if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 4:
                bgr = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2BGR)
            else:
                bgr = cropped_image

            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha_channel

            success = cv2.imwrite(save_path, bgra, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if success:
                self.progress.emit(100, "图像保存成功。")
                self.finished.emit(True, save_path, "")
            else:
                self.finished.emit(False, "", "写入 PNG 数据返回失败")

        except Exception as e:
            traceback.print_exc()
            self.finished.emit(False, "", f"保存抠图图像时发生错误: {e}")

    def _save_segmented_video(self, save_data: tuple, save_path: str, options: Optional[dict]):
        if options and options.get('output_format', '').lower() == 'gif':
            self._save_video_as_gif(save_data, save_path)
            return

        import os, uuid, tempfile, cv2, numpy as np, av, traceback, gc, subprocess, platform
        from fractions import Fraction

        (
            temp_frame_dir, processed_masks, total_frames,
            target_w, target_h, actual_save_fps,
            _, _, _,
            virtual_timeline
        ) = save_data

        codec_info = options.get('codec_info', {}) if options else {}
        global_export_transparent = bool(codec_info.get('alpha', False))

        encoder_name = codec_info.get('encoder', 'libx264')
        pix_fmt = codec_info.get('pix_fmt', 'yuv420p')

        if encoder_name == 'prores_ks' and global_export_transparent:
            pix_fmt = 'yuva444p10le'
        elif encoder_name == 'libx265' and 'yuva' in pix_fmt:
            pix_fmt = 'yuva420p'

        _, target_ext = os.path.splitext(save_path)
        ext_tmp = target_ext.lower() if target_ext else ".mp4"
        temp_silent_video = os.path.join(tempfile.gettempdir(), f"temp_silent_{uuid.uuid4().hex[:8]}{ext_tmp}")

        try:
            self.progress.emit(5, "正在解析混合时间线配置...")

            clip_configs = []
            current_count = 0
            for clip in virtual_timeline:
                bg_type = clip.get('bg_type')
                if bg_type is None:
                    if clip.get('bg_is_transparent', False):
                        bg_type = 'transparent'
                    elif clip.get('bg_image_path'):
                        bg_type = 'image'
                    elif clip.get('bg_color') is not None:
                        bg_type = 'color'
                    else:
                        bg_type = 'original'

                clip_configs.append({
                    'start': current_count,
                    'end': current_count + clip['frames'] - 1,
                    'bg_type': bg_type,
                    'bg_image_path': clip.get('bg_image_path', None),
                    'bg_color': clip.get('bg_color', QColor(0, 255, 0))
                })
                current_count += clip['frames']

            bg_canvas_cache = {}
            for idx, cfg in enumerate(clip_configs):
                path = cfg['bg_image_path']
                if cfg['bg_type'] == 'image' and path and os.path.exists(path):
                    bg_img = cv2.imread(path)
                    if bg_img is not None:
                        bg_canvas_cache[idx] = cv2.resize(bg_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            safe_rate = Fraction(actual_save_fps).limit_denominator(1000)

            self.progress.emit(10, f"初始化单路视频编码流 (格式={pix_fmt}, 编码器={encoder_name})...")

            silent_container = av.open(temp_silent_video, mode='w')
            silent_stream = silent_container.add_stream(encoder_name, rate=safe_rate)
            silent_stream.width = target_w
            silent_stream.height = target_h
            silent_stream.pix_fmt = pix_fmt

            if encoder_name == 'libx264':
                silent_stream.options = {'preset': 'medium', 'crf': '18', 'threads': 'auto'}
            elif encoder_name == 'libx265':
                silent_stream.options = {'preset': 'medium', 'crf': '23', 'threads': 'auto'}
            elif encoder_name == 'prores_ks':
                silent_stream.options = {
                    'profile': codec_info.get('profile', '4'),
                    'vendor': 'apl0',
                    'qscale': '9'
                }

            frame_global_idx = 0
            for clip_idx, cfg in enumerate(clip_configs):
                clip_bg_type = cfg['bg_type']

                for f_idx in range(cfg['start'], cfg['end'] + 1):
                    if self.is_cancelled:
                        raise InterruptedError("用户取消了导出。")

                    frame_path = os.path.join(temp_frame_dir, f"{f_idx:05d}.jpg")
                    if not os.path.exists(frame_path):
                        frame_path = os.path.join(temp_frame_dir, f"{f_idx}.jpg")

                    frame_cv = cv2.imread(frame_path)
                    if frame_cv is None:
                        frame_cv = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    if frame_cv.shape[:2] != (target_h, target_w):
                        frame_cv = cv2.resize(frame_cv, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                    frame_masks = processed_masks.get(f_idx, {})
                    h, w = frame_cv.shape[:2]

                    combined_alpha = np.zeros((h, w), dtype=np.float32)
                    has_mask = False

                    for mask_raw in frame_masks.values():
                        if mask_raw is not None:
                            mask_float = mask_raw.astype(np.float32) if mask_raw.dtype != bool else mask_raw.astype(np.float32)
                            if mask_float.shape != (h, w):
                                mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)
                            combined_alpha = np.maximum(combined_alpha, mask_float)
                            has_mask = True

                    # ---------------------------------------------------------
                    # 🚀 业界核心修复区：处理无蒙版对象的默认 Alpha 边界
                    # ---------------------------------------------------------
                    if global_export_transparent:
                        if clip_bg_type == 'transparent':
                            if has_mask:
                                alpha_channel = (combined_alpha * 255.0).clip(0, 255).astype(np.uint8)
                                alpha_3d = combined_alpha[:, :, np.newaxis]
                                frame_cv = (frame_cv.astype(np.float32) * alpha_3d).astype(np.uint8)
                            else:
                                # 修复：无蒙版等同于不透明，不能赋 0 抹除画面
                                alpha_channel = np.full((h, w), 255, dtype=np.uint8)
                        else:
                            if clip_bg_type == 'original':
                                pass
                            else:
                                if has_mask:
                                    if clip_idx in bg_canvas_cache:
                                        bg_canvas = bg_canvas_cache[clip_idx].copy()
                                    else:
                                        bg_bgr = (cfg['bg_color'].blue(), cfg['bg_color'].green(), cfg['bg_color'].red())
                                        bg_canvas = np.full((h, w, 3), bg_bgr, dtype=np.uint8)

                                    alpha_3d = combined_alpha[:, :, np.newaxis]
                                    blended_frame = frame_cv.astype(np.float32) * alpha_3d + bg_canvas.astype(np.float32) * (1.0 - alpha_3d)
                                    frame_cv = np.clip(blended_frame, 0.0, 255.0).astype(np.uint8)
                                else:
                                    # 修复：无蒙版时原视频作为全不透明前景完全遮盖背景，不做覆盖混合
                                    pass

                            alpha_channel = np.full((h, w), 255, dtype=np.uint8)

                        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                        frame_rgba = np.dstack((frame_rgb, alpha_channel))
                        av_frame = av.VideoFrame.from_ndarray(frame_rgba, format='rgba')
                    else:
                        if clip_bg_type == 'original':
                            pass
                        else:
                            if has_mask:
                                if clip_idx in bg_canvas_cache:
                                    bg_canvas = bg_canvas_cache[clip_idx].copy()
                                else:
                                    bg_bgr = (cfg['bg_color'].blue(), cfg['bg_color'].green(), cfg['bg_color'].red())
                                    bg_canvas = np.full((h, w, 3), bg_bgr, dtype=np.uint8)

                                alpha_3d = combined_alpha[:, :, np.newaxis]
                                blended_frame = frame_cv.astype(np.float32) * alpha_3d + bg_canvas.astype(np.float32) * (1.0 - alpha_3d)
                                frame_cv = np.clip(blended_frame, 0.0, 255.0).astype(np.uint8)
                            else:
                                # 修复：无蒙版时原生画面保留，拒绝纯色层穿透
                                pass

                        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')

                    av_frame.pts = frame_global_idx
                    frame_global_idx += 1

                    for packet in silent_stream.encode(av_frame):
                        silent_container.mux(packet)

                    pct = 10 + int((frame_global_idx / max(1, total_frames)) * 75)
                    self.progress.emit(pct, f"正在渲染合成视频帧 ({frame_global_idx}/{total_frames})...")

            for packet in silent_stream.encode():
                silent_container.mux(packet)
            silent_container.close()

            self.progress.emit(90, "正在混音并打包封装...")
            success, err_msg = self._merge_audio_with_pydub(temp_silent_video, virtual_timeline, save_path)

            if os.path.exists(temp_silent_video):
                os.remove(temp_silent_video)

            if success:
                self.progress.emit(100, "混合视频导出成功！")
                self.finished.emit(True, save_path, "")
            else:
                self.finished.emit(False, "", f"多音轨合成失败: {err_msg}")

        except Exception as e:
            if 'temp_silent_video' in locals() and os.path.exists(temp_silent_video):
                try:
                    os.remove(temp_silent_video)
                except Exception:
                    pass
            traceback.print_exc()
            self.finished.emit(False, "", f"混合视频编码崩溃: {e}")
        finally:
            gc.collect()

    def _merge_audio_with_pydub(self, temp_silent_video: str, virtual_timeline: list, save_path: str) -> tuple[bool, str]:
        """
        使用 Pydub 在内存中完成原声与BGM的完美防爆音混合。
        """
        import subprocess
        try:
            from pydub import AudioSegment
            import math
            import platform

            final_audio = AudioSegment.empty()

            for clip in virtual_timeline:
                fps = clip.get('fps', 30.0)
                dur_ms = int((clip.get('frames', 1) / fps) * 1000)
                in_ms = int((clip.get('in_point', 0) / fps) * 1000)

                orig_mute = clip.get('mute_all', False) or clip.get('mute_original', False)
                bgm_mute = clip.get('mute_all', False) or clip.get('mute_bgm', False)

                current_clip_audio = AudioSegment.silent(duration=dur_ms)
                if not orig_mute:
                    try:
                        orig = AudioSegment.from_file(clip['path'])
                        orig = orig[in_ms: in_ms + dur_ms]

                        vol = clip.get('original_audio_volume', 1.0)
                        if vol == 0:
                            orig = AudioSegment.silent(duration=dur_ms)
                        elif vol != 1.0:
                            orig = orig + (20 * math.log10(vol))

                        if len(orig) < dur_ms:
                            orig += AudioSegment.silent(duration=(dur_ms - len(orig)))
                        current_clip_audio = orig[:dur_ms]
                    except Exception as e:
                        print(f"Warning: 原声读取降级处理 {e}")

                bgm_path = clip.get('custom_audio_path')
                if bgm_path and os.path.exists(bgm_path) and not bgm_mute:
                    try:
                        bgm = AudioSegment.from_file(bgm_path)
                        bgm_start_ms = int(clip.get('custom_audio_clip_start', 0.0) * 1000)
                        bgm_end_ms = int(clip.get('custom_audio_clip_end', 0.0) * 1000)
                        if bgm_end_ms <= bgm_start_ms:
                            bgm_end_ms = len(bgm)

                        bgm = bgm[bgm_start_ms:bgm_end_ms]

                        bgm_vol = clip.get('custom_audio_volume', 1.0)
                        if bgm_vol == 0:
                            bgm = AudioSegment.silent(duration=len(bgm))
                        elif bgm_vol != 1.0:
                            bgm = bgm + (20 * math.log10(bgm_vol))

                        insert_ms = int(clip.get('custom_audio_start_sec', 0.0) * 1000)
                        current_clip_audio = current_clip_audio.overlay(bgm, position=insert_ms)
                    except Exception as e:
                        print(f"Warning: BGM 处理降级 {e}")

                final_audio += current_clip_audio

            import tempfile, uuid
            temp_wav = os.path.join(tempfile.gettempdir(), f"temp_mix_{uuid.uuid4().hex[:8]}.wav")
            final_audio.export(temp_wav, format="wav")

            self.progress.emit(92, "正在无损封装音视频流...")

            cmd = [
                'ffmpeg', '-y',
                '-i', temp_silent_video,
                '-i', temp_wav,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
                save_path
            ]

            startupinfo = None
            if platform.system() == "Windows":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo, check=True)

            if os.path.exists(temp_wav): os.remove(temp_wav)
            return True, ""

        except subprocess.CalledProcessError as e:
            err_details = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
            print(f"[FFmpeg Muxing Error]:\n{err_details}")
            if 'temp_wav' in locals() and os.path.exists(temp_wav):
                os.remove(temp_wav)
            return False, f"封装错误详情:\n{err_details}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            if 'temp_wav' in locals() and os.path.exists(temp_wav):
                os.remove(temp_wav)
            return False, str(e)

    def _merge_audio_ffmpeg(self, temp_silent_video: str, virtual_timeline: list, save_path: str) -> tuple[bool, str]:
        """FFmpeg C++ 后备极速封装"""
        inputs = ['-y', '-i', temp_silent_video]
        filter_nodes = []
        orig_audio_nodes = []
        bgm_audio_nodes = []

        current_input_idx = 1
        accum_dur = 0.0

        for i, clip in enumerate(virtual_timeline):
            fps = clip.get('fps', 30.0)
            in_sec = clip.get('in_point', 0) / fps
            dur_sec = clip.get('frames', 1) / fps

            inputs.extend(['-ss', f"{in_sec:.3f}", '-t', f"{dur_sec:.3f}", '-i', clip['path']])

            orig_mute = clip.get('mute_all', False) or clip.get('mute_original', False)
            vol = 0.0 if orig_mute else clip.get('original_audio_volume', 1.0)
            node_name = f"orig_{i}"

            filter_nodes.append(f"[{current_input_idx}:a]volume={vol:.3f}[{node_name}]")
            orig_audio_nodes.append(f"[{node_name}]")
            current_input_idx += 1

            bgm_path = clip.get('custom_audio_path')
            bgm_mute = clip.get('mute_all', False) or clip.get('mute_bgm', False)

            if bgm_path and os.path.exists(bgm_path) and not bgm_mute:
                bgm_start = clip.get('custom_audio_clip_start', 0.0)
                bgm_end = clip.get('custom_audio_clip_end', 0.0)
                bgm_dur = bgm_end - bgm_start
                if bgm_dur <= 0:
                    bgm_dur = 9999.0

                inputs.extend(['-ss', f"{bgm_start:.3f}", '-t', f"{bgm_dur:.3f}", '-i', bgm_path])

                delay_ms = int((accum_dur + clip.get('custom_audio_start_sec', 0.0)) * 1000)
                bgm_vol = clip.get('custom_audio_volume', 1.0)

                node_name_bgm = f"bgm_{i}"
                filter_nodes.append(
                    f"[{current_input_idx}:a]adelay={delay_ms}|{delay_ms},volume={bgm_vol:.3f}[{node_name_bgm}]"
                )
                bgm_audio_nodes.append(f"[{node_name_bgm}]")
                current_input_idx += 1

            accum_dur += dur_sec

        num_orig = len(orig_audio_nodes)
        mix_inputs = []
        if num_orig > 1:
            concat_filter = "".join(orig_audio_nodes) + f"concat=n={num_orig}:v=0:a=1[orig_full]"
            filter_nodes.append(concat_filter)
            mix_inputs.append("[orig_full]")
        elif num_orig == 1:
            filter_nodes.append(f"{orig_audio_nodes[0]}copy[orig_full]")
            mix_inputs.append("[orig_full]")

        mix_inputs.extend(bgm_audio_nodes)
        num_mix_inputs = len(mix_inputs)

        if num_mix_inputs > 1:
            mix_filter = "".join(
                mix_inputs) + f"amix=inputs={num_mix_inputs}:duration=first:dropout_transition=0[a_final]"
            filter_nodes.append(mix_filter)
            audio_map = "[a_final]"
        elif num_mix_inputs == 1:
            filter_nodes.append(f"{mix_inputs[0]}copy[a_final]")
            audio_map = "[a_final]"
        else:
            audio_map = None

        cmd = ['ffmpeg'] + inputs
        filter_complex_str = ";".join(filter_nodes)
        if filter_complex_str:
            cmd.extend(['-filter_complex', filter_complex_str])

        cmd.extend(['-map', '0:v'])
        if audio_map:
            cmd.extend(['-map', audio_map])
            cmd.extend(['-c:a', 'aac', '-b:a', '192k', '-ar', '44100'])
        else:
            cmd.extend(['-an'])

        cmd.extend(['-c:v', 'copy', save_path])

        try:
            import subprocess
            import platform
            startupinfo = None
            if platform.system() == "Windows":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            self.progress.emit(95, "正在完成多轨音视频极速封包...")
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo, text=True,
                           encoding='utf-8', check=True)
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, str(e.stderr)
        except FileNotFoundError:
            return False, "系统检测不到 FFmpeg 环境。"

    def _save_video_as_gif(self, save_data: tuple, save_path: str):
        """
        保存为动画 GIF：支持在单段视频中混合透明与非透明段。
        """
        from PIL import Image as PILImage
        import time
        import os
        import cv2
        import numpy as np
        import gc
        import concurrent.futures

        (
            temp_frame_dir, processed_masks, total_frames,
            target_w, target_h, actual_save_fps,
            _, _, _,
            virtual_timeline
        ) = save_data

        try:
            self.progress.emit(5, "启动 GIF 混合通道渲染引擎...")
            time.sleep(0.05)

            pil_frames = []
            frame_duration_ms = int(1000.0 / actual_save_fps)

            clip_configs = []
            current_count = 0
            for clip in virtual_timeline:
                clip_configs.append({
                    'start': current_count,
                    'end': current_count + clip['frames'] - 1,
                    'bg_is_transparent': clip.get('bg_is_transparent', False),
                    'bg_image_path': clip.get('bg_image_path', None),
                    'bg_color': clip.get('bg_color', QColor(0, 0, 0))
                })
                current_count += clip['frames']

            bg_canvas_cache = {}
            for idx, cfg in enumerate(clip_configs):
                path = cfg['bg_image_path']
                if not cfg['bg_is_transparent'] and path and os.path.exists(path):
                    bg_img = cv2.imread(path)
                    if bg_img is not None:
                        bg_canvas_cache[idx] = cv2.resize(bg_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            has_any_transparent_segment = any(cfg['bg_is_transparent'] for cfg in clip_configs)

            for i in range(total_frames):
                if self.is_cancelled:
                    if os.path.exists(save_path):
                        try:
                            os.remove(save_path)
                        except Exception:
                            pass
                    self.finished.emit(False, "", "用户取消了导出")
                    return

                frame_path = os.path.join(temp_frame_dir, f"{i:05d}.jpg")
                if not os.path.exists(frame_path):
                    frame_path = os.path.join(temp_frame_dir, f"{i}.jpg")

                frame_cv = cv2.imread(frame_path)
                if frame_cv is None:
                    frame_cv = np.zeros((target_h, target_w, 3), dtype=np.uint8)

                if frame_cv.shape[:2] != (target_h, target_w):
                    frame_cv = cv2.resize(frame_cv, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                frame_masks = processed_masks.get(i, {})
                h, w = frame_cv.shape[:2]

                combined_alpha = np.zeros((h, w), dtype=np.float32)
                has_mask = False

                for mask_raw in frame_masks.values():
                    if mask_raw is not None:
                        if mask_raw.dtype == bool:
                            mask_float = mask_raw.astype(np.float32)
                        else:
                            mask_float = np.clip(mask_raw.astype(np.float32), 0.0, 1.0)

                        if mask_float.shape != (h, w):
                            mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)
                        combined_alpha = np.maximum(combined_alpha, mask_float)
                        has_mask = True

                cfg_idx = -1
                active_cfg = None
                for idx, cfg in enumerate(clip_configs):
                    if cfg['start'] <= i <= cfg['end']:
                        active_cfg = cfg
                        cfg_idx = idx
                        break
                if active_cfg is None:
                    active_cfg = {'bg_is_transparent': False, 'bg_image_path': None, 'bg_color': QColor(0, 0, 0)}

                bg_is_transparent = active_cfg['bg_is_transparent']
                bg_color = active_cfg['bg_color']

                # ---------------------------------------------------------
                # 🚀 业界核心修复区：对齐 GIF Alpha 通道混色器行为规范
                # ---------------------------------------------------------
                if has_any_transparent_segment:
                    if bg_is_transparent:
                        bg_canvas = np.zeros((h, w, 4), dtype=np.uint8)
                        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                        bg_canvas[:, :, 0:3] = frame_rgb
                        if has_mask:
                            bg_canvas[:, :, 3] = (combined_alpha * 255.0).clip(0, 255).astype(np.uint8)
                        else:
                            # 修复：未遮罩剪辑保持物理状态的不透明完整展现
                            bg_canvas[:, :, 3] = 255
                        pil_img = PILImage.fromarray(bg_canvas, 'RGBA')
                    else:
                        if has_mask:
                            if cfg_idx in bg_canvas_cache:
                                bg_canvas_img = bg_canvas_cache[cfg_idx].copy()
                            else:
                                bg_bgr = (bg_color.blue(), bg_color.green(), bg_color.red())
                                bg_canvas_img = np.full((h, w, 3), bg_bgr, dtype=np.uint8)

                            alpha_3d = combined_alpha[:, :, np.newaxis]
                            blended_frame = frame_cv.astype(np.float32) * alpha_3d + bg_canvas_img.astype(np.float32) * (1.0 - alpha_3d)
                            frame_rgb = cv2.cvtColor(np.clip(blended_frame, 0.0, 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
                        else:
                            # 修复：防止纯色或图片将前景画面生吃强行抹掉
                            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)

                        rgba_canvas = np.zeros((h, w, 4), dtype=np.uint8)
                        rgba_canvas[:, :, 0:3] = frame_rgb
                        rgba_canvas[:, :, 3] = 255
                        pil_img = PILImage.fromarray(rgba_canvas, 'RGBA')
                else:
                    if has_mask:
                        if cfg_idx in bg_canvas_cache:
                            bg_canvas_img = bg_canvas_cache[cfg_idx].copy()
                        else:
                            bg_bgr = (bg_color.blue(), bg_color.green(), bg_color.red())
                            bg_canvas_img = np.full((h, w, 3), bg_bgr, dtype=np.uint8)

                        alpha_3d = combined_alpha[:, :, np.newaxis]
                        blended_frame = frame_cv.astype(np.float32) * alpha_3d + bg_canvas_img.astype(np.float32) * (1.0 - alpha_3d)
                        frame_rgb = cv2.cvtColor(np.clip(blended_frame, 0.0, 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    else:
                        # 修复：保持完全不透明的原始对象数据
                        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                    pil_img = PILImage.fromarray(frame_rgb, 'RGB')

                if has_any_transparent_segment:
                    pil_p = pil_img.convert('P', palette=PILImage.Palette.ADAPTIVE, colors=255)
                    alpha = pil_img.split()[3]
                    mask = alpha.point(lambda x: 255 if x < 128 else 0)
                    pil_p.paste(255, mask)
                else:
                    pil_p = pil_img.convert('P', palette=PILImage.Palette.ADAPTIVE, colors=256)

                pil_frames.append(pil_p)

                pct = 10 + int((i / max(1, total_frames)) * 75)
                self.progress.emit(pct, f"正在进行帧画面量化与调色板对齐... ({i + 1}/{total_frames})")

                time.sleep(0.01)

                if i % 15 == 0:
                    gc.collect()

            self.progress.emit(85, "调色板处理完成，正在封装并写入 GIF 文件...")
            time.sleep(0.1)

            if pil_frames:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    save_args = {
                        "save_all": True,
                        "append_images": pil_frames[1:],
                        "duration": frame_duration_ms,
                        "loop": 0,
                        "disposal": 2,
                        "optimize": False
                    }

                    if has_any_transparent_segment:
                        save_args["transparency"] = 255

                    future = executor.submit(
                        pil_frames[0].save,
                        save_path,
                        **save_args
                    )

                    fake_progress = 85
                    while not future.done():
                        if self.is_cancelled:
                            raise InterruptedError("用户取消了操作")

                        if fake_progress < 98:
                            fake_progress += 1
                            self.progress.emit(fake_progress, "正在将图像流持久化写入磁盘...")
                        time.sleep(0.1)

                    future.result()

            self.progress.emit(100, "GIF 动画保存成功！")
            self.finished.emit(True, save_path, "")

        except InterruptedError as ie:
            self.finished.emit(False, "", str(ie))
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(False, "", f"GIF 写入时发生未知异常: {e}")
        finally:
            if 'pil_frames' in locals():
                pil_frames.clear()
            gc.collect()
