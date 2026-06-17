import cv2
import numpy as np
import time
from typing import Optional


def refine_mask_with_matanyone2(
        mat_model,
        image_bgr_u8: np.ndarray,
        coarse_mask_u8: np.ndarray,
        erode_kernel_size: int = 10,
        dilate_kernel_size: int = 10,
        refine_iter: int = 10  # Default warmup frames (n_warmup) set to 10 to stabilize temporal features
) -> Optional[np.ndarray]:
    if mat_model is None or image_bgr_u8 is None or coarse_mask_u8 is None:
        return None

    t_start = time.time()

    try:
        import sys
        import torch
        import types
        from core.workers import PredictWorker
        from core.models.matanyone2.matanyone2_wrapper import matanyone2
        from core.models.matanyone2.inference.inference_core import InferenceCore

        # =====================================================================
        # 1. Retrieve the target hardware execution device
        # =====================================================================
        true_device_str = PredictWorker._get_true_target_device()
        target_device = torch.device(true_device_str)
        is_cpu_mode = (target_device.type == 'cpu')

        # =====================================================================
        # 2. Move model to hardware target and maximize CPU core allocation
        # =====================================================================
        mat_model = mat_model.to(target_device)
        if is_cpu_mode:
            mat_model = mat_model.float()

            import multiprocessing
            try:
                total_cores = multiprocessing.cpu_count()
                torch.set_num_threads(total_cores)
                print(f"CPU execution optimized: Forced activation of all {total_cores} threads for MatAnyone2 inference.")
            except Exception as e:
                print(f"Failed to configure multi-threaded CPU affinity: {e}")

        # =====================================================================
        # 3. Re-target submodule tensors to prevent cross-device reference leaks
        # =====================================================================
        for m in mat_model.modules():
            for k, v in vars(m).items():
                if isinstance(v, torch.Tensor):
                    setattr(m, k, v.to(target_device))

        # =====================================================================
        # 4. Update device parameters in the model config file
        # =====================================================================
        try:
            import omegaconf
            if hasattr(mat_model, 'cfg'):
                if omegaconf.OmegaConf.is_config(mat_model.cfg):
                    with omegaconf.open_dict(mat_model.cfg):
                        mat_model.cfg.device = target_device.type
                else:
                    mat_model.cfg.device = target_device.type
        except Exception:
            pass

        # =====================================================================
        # 5. System-level hook: Intercept global get_default_device lookups
        # =====================================================================
        overridden_modules = {}
        for mod_name, mod in list(sys.modules.items()):
            if 'matanyone2' in mod_name and hasattr(mod, 'get_default_device'):
                overridden_modules[mod] = getattr(mod, 'get_default_device')
                setattr(mod, 'get_default_device', lambda: target_device)

        try:
            image_rgb = cv2.cvtColor(image_bgr_u8, cv2.COLOR_BGR2RGB)
            frames = [image_rgb, image_rgb]

            template_mask = coarse_mask_u8.copy()
            if len(template_mask.shape) == 3:
                template_mask = template_mask[:, :, 0]

            # =================================================================
            # Preserve continuous alpha gradients (0-255 float range) instead
            # of hard binary thresholding.
            # =================================================================
            final_input_mask = template_mask.astype(np.float32)

            # Edge-case safeguard: Flip the first pixel if the mask is completely
            # uniform to prevent division-by-zero or attention collapse.
            if len(np.unique(final_input_mask)) == 1:
                final_input_mask[0, 0] = 255.0 if final_input_mask[0, 0] == 0.0 else 0.0

            with torch.inference_mode():
                processor = InferenceCore(mat_model, cfg=mat_model.cfg)
                processor.device = target_device

                # ==========================================================
                # 6. Target Device Hook: Cast inputs/outputs to target device
                # ==========================================================
                for attr_name in dir(processor):
                    attr = getattr(processor, attr_name)
                    if isinstance(attr, types.MethodType) and not attr_name.startswith("__"):
                        def make_hook(orig_func):
                            def hooked_func(*args, **kwargs):
                                new_args = [a.to(target_device) if isinstance(a, torch.Tensor) else a for a in args]
                                new_kwargs = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v) for k, v in
                                              kwargs.items()}
                                return orig_func(*new_args, **new_kwargs)

                            return hooked_func

                        setattr(processor, attr_name, make_hook(attr))

                # Mixed-precision inference execution
                with torch.autocast(device_type=target_device.type, enabled=not is_cpu_mode):
                    foreground, alpha_out = matanyone2(
                        processor,
                        frames,
                        final_input_mask,
                        r_erode=erode_kernel_size,
                        r_dilate=dilate_kernel_size,
                        n_warmup=refine_iter
                    )

                if target_device.type == 'cuda':
                    torch.cuda.synchronize(device=target_device)

            # =================================================================
            # Do not apply threshold cutoffs on edge boundaries.
            # Preserve delicate hair translucency details in raw output format.
            # =================================================================
            final_alpha = alpha_out[-1][:, :, 0].astype(np.float32) / 255.0

            print(f"--- MatAnyone 2 Inference Time: {time.time() - t_start:.4f}s | Warmup Iterations: {refine_iter} ---")
            return final_alpha

        finally:
            # Restore overridden hooks to prevent context pollution
            for mod, orig_func in overridden_modules.items():
                setattr(mod, 'get_default_device', orig_func)

    except Exception as e:
        print(f"MatAnyone 2 inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def refine_mask_with_matanyone2_tiled(
        mat_model,
        image_bgr_u8: np.ndarray,
        coarse_mask_u8: np.ndarray,
        tile_size: int = 1024,
        tile_pad: int = 64,
        erode_kernel_size: int = 10,
        dilate_kernel_size: int = 10,
        refine_iter: int = 10  # Default to 10 warmup frames under tiled execution mode
) -> Optional[np.ndarray]:
    if mat_model is None or image_bgr_u8 is None or coarse_mask_u8 is None:
        return None

    h, w = image_bgr_u8.shape[:2]
    THRESHOLD_720P = 1280

    # Execute standard full-image inference for dimensions below the threshold
    if max(h, w) <= THRESHOLD_720P:
        print(f"--- Image dimensions {w}x{h} do not exceed the {THRESHOLD_720P}px threshold. Executing full-frame inference ---")
        return refine_mask_with_matanyone2(
            mat_model, image_bgr_u8, coarse_mask_u8,
            erode_kernel_size, dilate_kernel_size, refine_iter
        )

    t_start = time.time()
    print(f"--- Activating memory-safe tiled processing: Dimensions {w}x{h} (>{THRESHOLD_720P}px) | Tile Size: {tile_size} ---")

    output_alpha = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    stride = tile_size

    processed_tiles = 0
    skipped_tiles = 0

    # High-performance tile split and merge calculations
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_start, y_end = y, min(y + stride, h)
            x_start, x_end = x, min(x + stride, w)

            py_start, py_end = max(0, y_start - tile_pad), min(h, y_end + tile_pad)
            px_start, px_end = max(0, x_start - tile_pad), min(w, x_end + tile_pad)

            img_tile = image_bgr_u8[py_start:py_end, px_start:px_end]
            mask_tile = coarse_mask_u8[py_start:py_end, px_start:px_end]

            unique_vals = np.unique(mask_tile)
            if len(unique_vals) == 1:
                # Skip redundant inference for uniform tiles (e.g., solid background)
                alpha_tile = (mask_tile.astype(np.float32) / 255.0)
                skipped_tiles += 1
            else:
                # Call the aligned single-tile refinement pipeline
                alpha_tile = refine_mask_with_matanyone2(
                    mat_model, img_tile, mask_tile,
                    erode_kernel_size, dilate_kernel_size, refine_iter
                )
                if alpha_tile is None:
                    alpha_tile = (mask_tile.astype(np.float32) / 255.0)
                processed_tiles += 1

            th, tw = alpha_tile.shape
            pt, pb = y_start - py_start, py_end - y_end
            pl, pr = x_start - px_start, px_end - x_end

            # Apply cosine linear feathering to blend overlapping boundaries and eliminate stitching artifacts
            weight = np.ones((th, tw), dtype=np.float32)
            if pt > 0: weight[:pt, :] *= np.linspace(0, 1, pt)[:, None]
            if pb > 0: weight[-pb:, :] *= np.linspace(1, 0, pb)[:, None]
            if pl > 0: weight[:, :pl] *= np.linspace(0, 1, pl)[None, :]
            if pr > 0: weight[:, -pr:] *= np.linspace(1, 0, pr)[None, :]

            output_alpha[py_start:py_end, px_start:px_end] += alpha_tile * weight
            weight_map[py_start:py_end, px_start:px_end] += weight

    safe_weight = np.where(weight_map == 0, 1.0, weight_map)
    final_alpha = output_alpha / safe_weight
    final_alpha = np.clip(final_alpha, 0.0, 1.0)

    print(
        f"--- Tiled inference complete: Elapsed {time.time() - t_start:.2f}s | Processed {processed_tiles} tiles | Skipped {skipped_tiles} tiles ---")
    return final_alpha