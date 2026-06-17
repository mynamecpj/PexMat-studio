# -*- coding: utf-8 -*-
import sys
import os
import shutil
import ctypes
import math
import av
import numpy as np

# ==========================================
# 全局无窗口进程拦截
# ==========================================
if sys.platform == "win32":
    import subprocess

    _original_popen = subprocess.Popen


    class PatchedPopen(_original_popen):
        def __init__(self, *args, **kwargs):
            # 强制注入无窗口创建标志，从根源杜绝一切第三方库调用引起的 CMD 黑框闪现
            creationflags = kwargs.get('creationflags', 0)
            creationflags |= 0x08000000  # CREATE_NO_WINDOW
            kwargs['creationflags'] = creationflags

            # 强行隐藏窗口句柄
            startupinfo = kwargs.get('startupinfo', None)
            if startupinfo is None:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            else:
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs['startupinfo'] = startupinfo
            super().__init__(*args, **kwargs)


    subprocess.Popen = PatchedPopen

from PySide6.QtWidgets import QApplication, QWidget, QLabel, QProgressBar
from PySide6.QtCore import (
    Qt, QTimer, QUrl, QThread, Signal, Slot
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# ==========================================
# 1. 独立流安全隔离
# ==========================================
class DummyStream:
    def write(self, data): pass

    def flush(self): pass

    def isatty(self): return False


def secure_console_io():
    if sys.stdout is None or not hasattr(sys.stdout, 'write'):
        sys.stdout = DummyStream()
    if sys.stderr is None or not hasattr(sys.stderr, 'write'):
        sys.stderr = DummyStream()


secure_console_io()


# ==========================================
# 2. 独立悬浮文字层
# ==========================================
class SplashOverlay(QWidget):
    def __init__(self, target_w, target_h, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedSize(target_w, target_h)

        self.text_label = QLabel("系统核心引擎准备中...", self)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF; 
                font-family: "Microsoft YaHei", sans-serif;
                font-size: 14px;
                font-weight: bold;
                background: transparent;
                border: none;
                text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.8);
            }
        """)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.2);
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #0A84FF;
                border-radius: 2px;
            }
        """)

        offset_from_bottom = max(30, int(target_h * 0.12))
        self.text_label.setGeometry(0, target_h - offset_from_bottom, target_w, 25)
        bar_w = int(target_w * 0.6)
        self.progress_bar.setGeometry((target_w - bar_w) // 2, target_h - offset_from_bottom + 30, bar_w, 4)


# ==========================================
# 3. 视频后台静默解压线程 (拒绝鼠标转圈)
# ==========================================
class VideoPreloadThread(QThread):
    """在后台线程完成所有耗时的 CPU 解码和缩放任务"""
    frames_ready = Signal(list, float)  # 传递 [QImage列表], fps

    def __init__(self, video_path, target_w, target_h):
        super().__init__()
        self.video_path = video_path
        self.target_w = target_w
        self.target_h = target_h

    def run(self):
        container = av.open(self.video_path)
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0

        images = []
        for frame in container.decode(video=0):
            img_array = frame.to_ndarray(format='rgba')
            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)

            h, w, c = img_array.shape

            qimg = QImage(img_array.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)

            # 在后台线程做缩放，彻底解放主线程
            qimg_scaled = qimg.scaled(
                self.target_w, self.target_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            images.append(qimg_scaled)

        container.close()
        self.frames_ready.emit(images, fps)


# ==========================================
# 4. 顶级真透明视频开屏模块
# ==========================================
class TransparentSplashScreen(QWidget):
    splash_ready_to_play = Signal()

    def __init__(self, video_path):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SplashScreen
        )

        # ------------------------------------------
        # 核心防黑屏属性：即使卡死也不渲染系统底色
        # ------------------------------------------
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background-color: transparent;")

        self.video_path = video_path
        self.frames_buffer = []
        self.total_frames = 0
        self.fps = 30.0

        # 预先获取视频分辨率并计算窗口大小 (此操作耗时极短，可放主线程)
        container = av.open(self.video_path)
        video_stream = container.streams.video[0]
        vid_w = video_stream.width
        vid_h = video_stream.height
        container.close()

        aspect_ratio = vid_w / vid_h if vid_h > 0 else 16 / 9
        screen = QApplication.primaryScreen().geometry()
        target_area = screen.width() * screen.height() * 0.30

        self.splash_height = int(math.sqrt(target_area / aspect_ratio))
        self.splash_width = int(self.splash_height * aspect_ratio)

        if self.splash_width > screen.width() * 0.8:
            self.splash_width = int(screen.width() * 0.8)
            self.splash_height = int(self.splash_width / aspect_ratio)

        self.setFixedSize(self.splash_width, self.splash_height)

        # UI组件挂载
        self.video_label = QLabel(self)
        self.video_label.setGeometry(0, 0, self.splash_width, self.splash_height)
        self.video_label.setStyleSheet("background-color: transparent;")

        self.overlay = SplashOverlay(self.splash_width, self.splash_height, self)
        self.overlay.setGeometry(0, 0, self.splash_width, self.splash_height)

        # 音频装载
        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(1.0)
        self.audio_player = QMediaPlayer()
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_player.setSource(QUrl.fromLocalFile(self.video_path))
        self.audio_player.mediaStatusChanged.connect(self._on_media_status_changed)

        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(self._sync_frame_to_audio)

        # 启动后台解码线程
        self.preloader = VideoPreloadThread(self.video_path, self.splash_width, self.splash_height)
        self.preloader.frames_ready.connect(self._on_frames_ready)
        self.preloader.start()

    @Slot(list, float)
    def _on_frames_ready(self, images_list, fps):
        """后台解压完毕，瞬间将 QImage 转为 GPU 友好的 QPixmap"""
        self.fps = fps
        self.frames_buffer = [QPixmap.fromImage(img) for img in images_list]
        self.total_frames = len(self.frames_buffer)

        if self.total_frames > 0:
            self.video_label.setPixmap(self.frames_buffer[0])

        self.splash_ready_to_play.emit()

    def _on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.audio_player.setPosition(0)
            self.audio_player.play()

    def _sync_frame_to_audio(self):
        """主时钟对齐机制"""
        if self.total_frames == 0:
            return

        audio_pos_ms = self.audio_player.position()
        target_frame_idx = int((audio_pos_ms / 1000.0) * self.fps)
        target_frame_idx = target_frame_idx % self.total_frames

        self.video_label.setPixmap(self.frames_buffer[target_frame_idx])

    def center_on_screen(self):
        screen_rect = QApplication.primaryScreen().geometry()
        x = (screen_rect.width() - self.width()) // 2
        y = (screen_rect.height() - self.height()) // 2
        self.move(x, y)

    def trigger_engine_start(self):
        self.show()
        self.audio_player.play()
        self.sync_timer.start(16)

    def update_progress(self, text, value):
        self.overlay.text_label.setText(text)
        self.overlay.progress_bar.setValue(value)
        QApplication.processEvents()

    def closeEvent(self, event):
        self.sync_timer.stop()
        if hasattr(self, 'audio_player'):
            self.audio_player.stop()
        super().closeEvent(event)


# ==========================================
# 5. FFmpeg / Pydub 音频环境高秒开优化
# ==========================================
def configure_ffmpeg_for_pydub():
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
    else:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))

    # 【修复点 1】：精确定位到 ffmpeg 子文件夹
    ffmpeg_dir = os.path.join(bundle_dir, "ffmpeg")

    # 【修复点 2】：将 ffmpeg 子文件夹的环境变量置于 PATH 最前
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    try:
        from pydub import AudioSegment
        import pydub.utils
        import pydub.audio_segment
        import json
        import subprocess

        # 【修复点 3】：指向子文件夹内的可执行文件
        ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        ffprobe_exe = os.path.join(ffmpeg_dir, "ffprobe.exe")

        def patched_which(program):
            if "ffmpeg" in program:
                return ffmpeg_exe if os.path.exists(ffmpeg_exe) else None
            if "ffprobe" in program:
                return ffprobe_exe if os.path.exists(ffprobe_exe) else None
            # Pydub 会尝试找 avconv，直接返回 None，让它死心，绝不扫描硬盘
            return None

        pydub.utils.which = patched_which

        if os.path.exists(ffmpeg_exe):
            AudioSegment.converter = ffmpeg_exe
            pydub.utils.get_prober_name = lambda: ffprobe_exe
            pydub.audio_segment.get_prober_name = lambda: ffprobe_exe

        def patched_mediainfo_json(filepath, *args, **kwargs):
            safe_path = filepath
            if os.name == 'nt' and isinstance(filepath, str):
                buf = ctypes.create_unicode_buffer(1024)
                ctypes.windll.kernel32.GetShortPathNameW(filepath, buf, 1024)
                if buf.value:
                    safe_path = buf.value

            prober_path = ffprobe_exe if os.path.exists(ffprobe_exe) else "ffprobe"

            cmd = [
                prober_path, "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", safe_path
            ]
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = 0x08000000  # CREATE_NO_WINDOW

            try:
                result = subprocess.run(
                    cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding='utf-8',
                    errors='ignore', startupinfo=startupinfo,
                    creationflags=creationflags, timeout=10
                )
                stdout_text = result.stdout.strip()
                if not stdout_text:
                    raise RuntimeError("ffprobe stdout 为空。")
                return json.loads(stdout_text)
            except Exception as e:
                raise e

        pydub.utils.mediainfo_json = patched_mediainfo_json
        pydub.audio_segment.mediainfo_json = patched_mediainfo_json

    except ImportError:
        pass


# ==========================================
# 6. 主程序入口
# ==========================================
def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)


    from config.settings import TEMP_BASE_DIR
    if not os.path.exists(TEMP_BASE_DIR):
        os.makedirs(TEMP_BASE_DIR, exist_ok=True)

    app = QApplication(sys.argv)

    from ui.style import apply_stylesheet
    apply_stylesheet(app)

    media_filename = "splash.mov"

    if getattr(sys, 'frozen', False):
        media_path = os.path.join(sys._MEIPASS, "assets/open_asset", media_filename)
    else:
        media_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets/open_asset", media_filename)

    my_splash = TransparentSplashScreen(media_path)
    my_splash.center_on_screen()

    progress_val = 0
    progress_timer = QTimer()

    def update_fake_progress():
        nonlocal progress_val
        if progress_val < 95:
            progress_val += 1
            if progress_val < 30:
                my_splash.update_progress("正在连接硬件加速模块...", progress_val)
            elif progress_val < 70:
                my_splash.update_progress("正在装载 AI 图像与视频模型...", progress_val)
            else:
                my_splash.update_progress("正在初始化后台并行计算引擎...", progress_val)

    progress_timer.timeout.connect(update_fake_progress)

    def start_heavy_backend():
        # 在开屏动画已经流畅播放时，再在后台瞬间初始化 FFmpeg 环境
        configure_ffmpeg_for_pydub()

        progress_timer.start(50)
        from core.workers import HeadlessLoader
        global loader
        loader = HeadlessLoader()
        loader.loading_complete.connect(on_loading_complete, Qt.ConnectionType.QueuedConnection)
        loader.start()

    def on_loading_complete(image_pred, video_pred, mat_model):
        progress_timer.stop()
        my_splash.update_progress("引擎启动完成，正在进入工作区！", 100)
        QTimer.singleShot(100, lambda: launch_main_window(image_pred, video_pred, mat_model))

    def launch_main_window(image_pred, video_pred, mat_model):
        global main_window
        from ui.main_window import ImageEnhancerApp
        main_window = ImageEnhancerApp(image_pred, video_pred, mat_model)
        main_window.show()
        my_splash.close()

    def ignition():
        my_splash.trigger_engine_start()
        # 稍微给视频留出 250ms 的播放缓冲，再启动重度加载
        QTimer.singleShot(250, start_heavy_backend)

    my_splash.splash_ready_to_play.connect(ignition)

    ret_code = app.exec()

    if os.path.exists(TEMP_BASE_DIR):
        shutil.rmtree(TEMP_BASE_DIR, ignore_errors=True)

    sys.exit(ret_code)


if __name__ == '__main__':
    main()