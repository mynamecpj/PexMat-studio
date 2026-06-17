# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_all

sys.setrecursionlimit(5000)
block_cipher = None

project_root = os.path.abspath(os.path.dirname(spec_val) if 'spec_val' in locals() else os.getcwd())

def find_local_submodules(root_dir, package_name):
    submodules = []
    package_path = os.path.join(root_dir, package_name)
    if not os.path.exists(package_path):
        return submodules
    for root, dirs, files in os.walk(package_path):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                mod_name = os.path.splitext(rel_path)[0].replace(os.sep, '.')
                submodules.append(mod_name)
        for d in dirs:
            init_file = os.path.join(root, d, '__init__.py')
            if os.path.exists(init_file):
                rel_path = os.path.relpath(os.path.join(root, d), root_dir)
                mod_name = rel_path.replace(os.sep, '.')
                submodules.append(mod_name)
    return list(set(submodules))

import torch
torch_onnx_dir = os.path.join(os.path.dirname(torch.__file__), 'onnx')

custom_binaries = []
ffmpeg_local_dir = os.path.join(project_root, 'ffmpeg')
if os.path.exists(ffmpeg_local_dir):
    custom_binaries += [
        (os.path.join(ffmpeg_local_dir, 'ffmpeg.exe'), '.'),
        (os.path.join(ffmpeg_local_dir, 'ffprobe.exe'), '.'),
    ]

mpv_dll_path = os.path.join(project_root, 'libmpv-2.dll')
if os.path.exists(mpv_dll_path):
    custom_binaries.append((mpv_dll_path, '.'))

app_datas = [
    ('assets/app_icon.ico', '.'),
    ('assets', 'assets'),
    ('sam2/configs', 'sam2/configs'),
    ('MEMatte/configs', 'MEMatte/configs'),
    ('checkpoints', 'checkpoints'),
    (torch_onnx_dir, 'torch/onnx'),
    ('core/models/matanyone2/config', 'core/models/matanyone2/config'),
]

hidden_imports = [
    'ui', 'ui.main_window', 'core', 'core.workers', 'config', 'config.settings',
    'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'PySide6.QtMultimedia',
    'torch', 'torchvision', 'torchvision.ops', 'torchvision.transforms',
    'sympy', 'networkx', 'safetensors',
    'ultralytics', 'detectron2', 'fvcore', 'iopath', 'hydra', 'omegaconf', 'sam2',
    'numpy', 'cv2', 'PIL', 'PIL.Image', 'PIL.ImageSequence', 'scipy',
    'uuid', 'ctypes', 'shutil', 'json', 'traceback', 'threading', 'time', 'collections', 'multiprocessing',
    'lap', 'mpv', 'av', 'pydub'
]

hidden_imports += find_local_submodules(project_root, 'MEMatte')

packages_to_collect = [
    'transformers', 'huggingface_hub', 'timm', 'einops', 'accelerate',
    'tokenizers', 'safetensors', 'regex', 'basicsr', 'realesrgan',
    'kornia', 'detectron2', 'fvcore', 'mpv', 'av'
]

for pkg in packages_to_collect:
    datas, binaries, hidden = collect_all(pkg)
    app_datas += datas
    custom_binaries += binaries
    hidden_imports += hidden

a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=custom_binaries,
    datas=app_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch.onnx', 'onnx.reference', 'onnxscript', 'onnx', 'onnxruntime',
        'tensorboard', 'torch.tensorboard', 'triton', 'tkinter', 'matplotlib',
        'pandas', 'notebook', 'ipython', 'PyQt5', 'PyQt6', 'sklearn', 'scikit-learn'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode={
        'sam2': 'py', 'MEMatte': 'py', 'hydra': 'py', 'transformers': 'py',
        'huggingface_hub': 'py', 'ultralytics': 'py', 'timm': 'py', 'einops': 'py',
        'basicsr': 'py', 'realesrgan': 'py', 'kornia': 'py', 'mpv': 'py'
    }
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 【核心修改】：去掉了 PyInstaller 的 Splash，完全用纯 Python 解决！
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PexMat-Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # <--- 保持 False，提升启动速度
    console=False,      # <--- 隐藏背后的黑框 CMD
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app_icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,          # <--- 保持 False
    upx_exclude=[],
    name='PexMat-Studio'
)