import contextlib
import torch
import functools


def get_default_device():
    """
    【全局终极接管】：穿透读取主程序的设置，坚决摒弃盲目的硬件探测！
    """
    try:
        # 直接读取用户在软件界面的 CPU/GPU 切换开关，指哪打哪
        from core.workers import PredictWorker
        true_device_str = PredictWorker._get_true_target_device()
        return torch.device(true_device_str)
    except Exception:
        # 如果获取失败（兜底逻辑），再退回原来的物理探测
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def safe_autocast_decorator(enabled=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device = get_default_device()

            # 【核心防御 1】：如果是 CPU 模式，坚决彻底禁用半精度 (autocast)！
            # 否则必定报 FloatTensor 和 HalfTensor 的数据类型冲突
            actual_enabled = enabled if device.type == "cuda" else False

            if device.type in ["cuda", "cpu"]:
                with torch.amp.autocast(device_type=device.type, enabled=actual_enabled):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextlib.contextmanager
def safe_autocast(enabled=True):
    device = get_default_device()

    # 【核心防御 2】：如果是 CPU 模式，坚决彻底禁用半精度
    actual_enabled = enabled if device.type == "cuda" else False

    if device.type in ["cuda", "cpu"]:
        with torch.amp.autocast(device_type=device.type, enabled=actual_enabled):
            yield
    else:
        yield  # MPS or other unsupported backends skip autocast