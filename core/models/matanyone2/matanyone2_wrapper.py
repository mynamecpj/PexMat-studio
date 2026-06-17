import time
import tqdm
import torch
from torchvision.transforms.functional import to_tensor
import numpy as np
import random
import cv2
from core.models.matanyone2.utils.device import get_default_device, safe_autocast_decorator

# Global variable retained solely for legacy compatibility with deprecated code modules.
# Modern operations should access devices via context-specific attributes.
device = get_default_device()


def gen_dilate(alpha: np.ndarray, min_kernel_size: int, max_kernel_size: int) -> np.ndarray:
    """
    Applies morphological dilation using an elliptical structuring element of random size.

    Args:
        alpha (np.ndarray): Input alpha matte channel.
        min_kernel_size (int): Minimum bounds of the elliptical kernel diameter.
        max_kernel_size (int): Maximum bounds of the elliptical kernel diameter.

    Returns:
        np.ndarray: Dilated mask of type float32.
    """
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1) * 255
    return dilate.astype(np.float32)


def gen_erosion(alpha: np.ndarray, min_kernel_size: int, max_kernel_size: int) -> np.ndarray:
    """
    Applies morphological erosion using an elliptical structuring element of random size.

    Args:
        alpha (np.ndarray): Input alpha matte channel.
        min_kernel_size (int): Minimum bounds of the elliptical kernel diameter.
        max_kernel_size (int): Maximum bounds of the elliptical kernel diameter.

    Returns:
        np.ndarray: Eroded mask of type float32.
    """
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1) * 255
    return erode.astype(np.float32)


@torch.inference_mode()
@safe_autocast_decorator()
def matanyone2(processor, frames_np, mask, r_erode=0, r_dilate=0, n_warmup=10, progress_callback=None):
    """
    Core inference pipeline for high-precision temporal matting.
    Optimized to bypass stdout-blocking CLI loops (tqdm) in favor of GUI-friendly progress callbacks.

    Args:
        processor: Instance of the InferenceCore model processor.
        frames_np (list of np.ndarray): Sequence of raw input frames in uint8 (H, W, C) format.
        mask (np.ndarray): Initial binary segmentation mask (H, W) in uint8 format.
        r_erode (int): Morphological erosion radius.
        r_dilate (int): Morphological dilation radius.
        n_warmup (int): Number of stabilization warmup frames to prepend.
        progress_callback (callable, optional): Signature: callback(current_index, total_count).

    Returns:
        tuple: (list of composite frames, list of alpha masks)
    """
    curr_device = getattr(processor, 'device', get_default_device())

    # Default chroma-key green compositing background
    bgr = (np.array([120, 255, 155], dtype=np.float32) / 255).reshape((1, 1, 3))
    objects = [1]

    # Apply morphological preprocessing to clean up mask edges
    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    mask = torch.from_numpy(mask).to(curr_device)

    # Pad with initial frame to stabilize temporal context warm-up
    frames_np = [frames_np[0]] * n_warmup + frames_np

    frames = []
    phas = []
    total_frames = len(frames_np)

    for ti, frame_single in enumerate(frames_np):
        image = to_tensor(frame_single).float().to(curr_device)

        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)  # Encode initial frame spatial features
            output_prob = processor.step(image, first_frame_pred=True)  # Clear initial queue references
        else:
            if ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True)
            else:
                output_prob = processor.step(image)

        # Map probabilistic logits to structural alpha matte
        mask_out = processor.output_prob_to_mask(output_prob)
        pha = mask_out.unsqueeze(2).detach().to("cpu").numpy()
        com_np = frame_single / 255. * pha + bgr * (1 - pha)

        # Filter out output matrices generated during initial temporal warmup phase
        if ti > (n_warmup - 1):
            frames.append((com_np * 255).astype(np.uint8))
            phas.append((pha * 255).astype(np.uint8))

        # Push execution progress to the active UI handler
        if progress_callback is not None:
            progress_callback(ti, total_frames)

        # Synchronize execution streams to prevent micro-stuttering in OS window compositing
        if curr_device.type == 'cuda':
            torch.cuda.synchronize(device=curr_device)

        # Relinquish execution context briefly to allow GIL handling inside multithreaded runtimes
        time.sleep(0.015)

    return frames, phas