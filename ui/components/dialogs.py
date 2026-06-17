import os
from pathlib import Path

from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QProgressDialog, QDialog, QVBoxLayout, QWidget, QGraphicsDropShadowEffect,
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QFrame
)
from PySide6.QtCore import Qt, QSize, Signal, QEvent

from config.settings import _TR


class ProgressDialog(QProgressDialog):
    """Modern dark-themed progress dialog."""

    def __init__(self, title, parent=None, cancel_text="取消"):
        # Translate parameters immediately during instantiation
        translated_title = _TR(title)
        translated_cancel = _TR(cancel_text)

        super().__init__(translated_title, translated_cancel, 0, 100, parent)
        self.setWindowTitle(_TR("处理中"))
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setAutoClose(True)
        self.setAutoReset(True)
        self.setMinimumDuration(100)
        self.setFixedSize(360, 140)

        self.setStyleSheet("""
            QProgressDialog {
                background-color: #262626;
                color: #E0E0E0;
                border: 1px solid #333333;
                border-radius: 8px;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 14px;
                font-weight: bold;
                margin-top: 10px;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                text-align: center;
                background-color: #1A1A1A;
                color: #FFFFFF;
                height: 12px;
                margin-top: 10px;
                margin-bottom: 10px;
            }
            QProgressBar::chunk {
                background-color: #1A73E8;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #333333;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 6px 20px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)


class DropdownOverlay(QFrame):
    """
    仿 剪映/CapCut 机制的内部覆盖菜单图层
    作为一个完全属于 Dialog 的子部件进行渲染，彻底杜绝多窗口 DWM 合成时的层级闪烁。
    """
    selected = Signal(int)

    def __init__(self, items, current_index, combo_box, parent):
        super().__init__(parent)
        self.combo_box = combo_box
        self.setObjectName("DropdownOverlay")

        # 匹配卡片的高级暗黑主题样式
        self.setStyleSheet("""
            QFrame#DropdownOverlay {
                background-color: #2C2C2E;
                border: 1px solid #444444;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # 动态创建平铺的悬浮按钮作为下拉子项
        for i, item in enumerate(items):
            btn = QPushButton(item)
            btn.setFixedHeight(34)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

            if i == current_index:
                # 蓝色高亮选中项
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #0A84FF;
                        color: #FFFFFF;
                        text-align: left;
                        padding: 6px 12px;
                        border: none;
                        border-radius: 6px;
                        font-size: 13px;
                        font-weight: bold;
                    }
                """)
            else:
                # 默认暗色项
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        color: #E0E0E0;
                        text-align: left;
                        padding: 6px 12px;
                        border: none;
                        border-radius: 6px;
                        font-size: 13px;
                    }
                    QPushButton:hover {
                        background-color: #3A3A3C;
                        color: #FFFFFF;
                    }
                """)
            # 使用默认参数绑定索引
            btn.clicked.connect(lambda checked=False, index=i: self.selected.emit(index))
            layout.addWidget(btn)

        # 在父对话框上安装事件过滤器，用于捕获外部点击，从而关闭菜单
        self.parent().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            pos = event.pos()
            # 将点击坐标转换为全局，再映射到本下拉菜单内
            global_pos = self.parent().mapToGlobal(pos)

            # 1. 如果点击在下拉菜单内部，不作处理
            local_pos = self.mapFromGlobal(global_pos)
            if self.rect().contains(local_pos):
                return super().eventFilter(obj, event)

            # 2. 如果点击发生在触发此下拉的组合框上，协助关闭并拦截，防止二次触发开启
            combo_local_pos = self.combo_box.mapFromGlobal(global_pos)
            if self.combo_box.rect().contains(combo_local_pos):
                self.close()
                return True

                # 3. 点击外部区域，直接关闭下拉栏
            self.close()

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self.parent().removeEventFilter(self)
        super().closeEvent(event)


class OverlayComboBox(QFrame):
    """
    纯内部渲染的专业级下拉组合框
    用作原 QComboBox 的完全替代品，100% 免疫 Windows 窗口穿透与闪烁。
    """
    currentIndexChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("OverlayComboBox")
        self.items = []
        self._current_index = -1
        self._menu = None

        # 内部布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(0)

        self.label = QLabel("")
        self.label.setStyleSheet("color: #FFFFFF; font-size: 14px; background: transparent;")

        self.arrow = QLabel("▼")
        self.arrow.setStyleSheet("color: #888888; font-size: 10px; background: transparent;")
        self.arrow.setFixedWidth(15)
        self.arrow.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.label, 1)
        layout.addWidget(self.arrow)

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet("""
            QFrame#OverlayComboBox {
                background-color: #1F1F1F;
                border: 1px solid #3A3A3C;
                border-radius: 8px;
            }
            QFrame#OverlayComboBox:hover {
                border: 1px solid #555555;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.show_dropdown()

    def addItems(self, items):
        self.items.extend(items)
        if self._current_index == -1 and items:
            self.setCurrentIndex(0)

    def currentText(self):
        if 0 <= self._current_index < len(self.items):
            return self.items[self._current_index]
        return ""

    def setCurrentText(self, text):
        if text in self.items:
            self.setCurrentIndex(self.items.index(text))

    def setCurrentIndex(self, index):
        if 0 <= index < len(self.items):
            self._current_index = index
            self.label.setText(self.items[index])
            self.currentIndexChanged.emit(index)

    def show_dropdown(self):
        # 如果已经打开，再次点击则关闭
        if self._menu and self._menu.isVisible():
            self._menu.close()
            return

        top_window = self.window()
        if not top_window:
            return

        # 创建内部覆盖式下拉面板，直接挂载到 Dialog 窗口上
        self._menu = DropdownOverlay(self.items, self._current_index, self, top_window)

        # 核心算法：计算渲染位置
        # 获取触发按钮在 Dialog 中的坐标
        pos = self.mapTo(top_window, self.rect().bottomLeft())
        menu_height = len(self.items) * 36 + 10

        # 防溢出与自适应：如果下方空间不足（会盖住按钮），则自动向上弹出
        if pos.y() + menu_height > top_window.height() - 20:
            top_pos = self.mapTo(top_window, self.rect().topLeft())
            self._menu.setGeometry(top_pos.x(), top_pos.y() - menu_height - 4, self.width(), menu_height)
        else:
            # 正常向下展开
            self._menu.setGeometry(pos.x(), pos.y() + 4, self.width(), menu_height)

        self._menu.selected.connect(self.on_item_selected)
        self._menu.show()

        # 强制将渲染图层提升至主窗口的最上方，实现 100% 遮挡所有按钮
        self._menu.raise_()

    def on_item_selected(self, index):
        self.setCurrentIndex(index)
        if self._menu:
            self._menu.close()


class ModernExportDialog(QDialog):
    """
    导出设置窗口
    """

    def __init__(self, export_type, original_size, default_name, default_fps=30.0, has_transparent_bg=False,
                 parent=None):
        super().__init__(parent)
        self.export_type = export_type
        self.original_size = original_size
        self.aspect_ratio = original_size.width() / max(1, original_size.height())
        self.default_name = default_name
        self.default_fps = default_fps
        self.has_transparent_bg = has_transparent_bg
        self.target_size = original_size

        self.CODECS_MP4 = {
            "H.264 / AVC (高兼容性, 无透明)": {"encoder": "libx264", "pix_fmt": "yuv420p", "alpha": False},
            "H.265 / HEVC (高压缩比, 无透明)": {"encoder": "libx265", "pix_fmt": "yuv420p", "alpha": False},
            "AV1 (极高压缩比, 无透明)": {"encoder": "libsvtav1", "pix_fmt": "yuv420p", "alpha": False},
        }

        self.CODECS_MOV = {
            "ProRes 4444 (支持透明)": {"encoder": "prores_ks", "profile": "4", "pix_fmt": "yuva444p10le",
                                                   "alpha": True},
            "PNG 序列编码 (支持透明兼容性强)": {"encoder": "png", "pix_fmt": "rgba", "alpha": True},
            "H.265 / HEVC (高压缩比, 无透明)": {"encoder": "libx265", "pix_fmt": "yuv420p", "alpha": False},
            "H.264 / AVC (高兼容性, 无透明)": {"encoder": "libx264", "pix_fmt": "yuv420p", "alpha": False},
        }

        self.CODECS_AVI = {
            "H.264 / AVC (高兼容性, 无透明)": {"encoder": "libx264", "pix_fmt": "yuv420p", "alpha": False},
            "MPEG-4 (传统AVI编码画质差, 无透明)": {"encoder": "mpeg4", "pix_fmt": "yuv420p", "alpha": False},
        }

        self.setWindowTitle("导出设置" if export_type != 'video' else "导出视频")
        self.setMinimumSize(600, 580 if export_type == 'video' else 480)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._setup_ui()

        if self.export_type == 'video':
            self.format_combo.items = ["MP4", "MOV", "GIF", "AVI"]
            self.format_combo.setCurrentIndex(0)

        self._init_data()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        bg_widget = QWidget()
        bg_widget.setStyleSheet("""
            QWidget#MainBG { background-color: #181818; border-radius: 16px; border: 1px solid #333333; }
            QLabel { color: #E0E0E0; font-family: -apple-system, "Microsoft YaHei", sans-serif; font-size: 14px; }
        """)
        bg_widget.setObjectName("MainBG")

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(0, 8)
        bg_widget.setGraphicsEffect(shadow)

        layout = QVBoxLayout(bg_widget)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)

        title_layout = QHBoxLayout()
        title_lbl = QLabel("导出" if self.export_type != 'video' else "导出视频")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF; letter-spacing: 1px;")

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(32, 32)
        close_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        close_btn.setStyleSheet("""
            QPushButton { background: transparent; border: none; font-size: 16px; color: #888888; border-radius: 16px; }
            QPushButton:hover { background: #333333; color: #FFFFFF; }
        """)
        close_btn.clicked.connect(self.reject)

        title_layout.addWidget(title_lbl)
        title_layout.addStretch()
        title_layout.addWidget(close_btn)
        layout.addLayout(title_layout)

        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #2A2A2A;")
        layout.addWidget(line)

        global_control_style = """
            QLineEdit { background-color: #1F1F1F; border: 1px solid #3A3A3C; border-radius: 8px; padding: 10px 14px; color: #FFFFFF; font-size: 14px; }
            QLineEdit:focus { border: 1px solid #0A84FF; background-color: #242424; }
        """

        card_widget = QWidget()
        card_widget.setStyleSheet(
            global_control_style + "QWidget#CardBG { background-color: #242424; border-radius: 12px; }")
        card_widget.setObjectName("CardBG")
        card_layout = QVBoxLayout(card_widget)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(16)

        def create_form_row(label_text, widget_obj):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(90)
            lbl.setStyleSheet("color: #A0A0A5; font-weight: bold; font-size: 14px;")
            row.addWidget(lbl)
            row.addWidget(widget_obj, 1)
            return row

        self.name_input = QLineEdit()
        card_layout.addLayout(create_form_row("导出名称", self.name_input))

        path_container = QWidget()
        path_layout = QHBoxLayout(path_container)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(10)

        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        self.path_input.setStyleSheet(self.path_input.styleSheet() + "color: #A0A0A5;")

        browse_btn = QPushButton("📁")
        browse_btn.setFixedSize(40, 40)
        browse_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        browse_btn.setStyleSheet(
            "QPushButton { background-color: #333333; border-radius: 8px; border: 1px solid #404040; font-size: 16px;} QPushButton:hover { background-color: #404040; border-color: #555555; }")
        browse_btn.clicked.connect(self._browse_path)

        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_btn)
        card_layout.addLayout(create_form_row("导出位置", path_container))

        res_container = QWidget()
        res_layout = QHBoxLayout(res_container)
        res_layout.setContentsMargins(0, 0, 0, 0)

        self.res_combo = OverlayComboBox()
        self.res_combo.addItems(["原画 (Original)", "4K (2160P)", "2K (1440P)", "1080P", "720P"])
        self.res_combo.currentIndexChanged.connect(self._on_res_changed)

        self.size_info_lbl = QLabel("")
        self.size_info_lbl.setStyleSheet("color: #6B7280; font-size: 13px; margin-left: 10px;")

        res_layout.addWidget(self.res_combo, 1)
        res_layout.addWidget(self.size_info_lbl)
        card_layout.addLayout(create_form_row("分辨率", res_container))

        # --- 格式选择 ---
        self.format_combo = OverlayComboBox()
        if self.export_type == 'video':
            self.format_combo.addItems(["MP4", "MOV", "GIF", "AVI"])
        else:
            self.format_combo.addItems(["PNG", "JPG"])
        card_layout.addLayout(create_form_row("格式", self.format_combo))

        # --- 【新增UI重构部分】：直接放置在格式下方、帧率上方 ---
        if self.export_type == 'video':
            self.codec_combo = OverlayComboBox()
            card_layout.addLayout(create_form_row("视频编码", self.codec_combo))

            self.warning_label = QLabel("⚠️ 当前编码不支持 Alpha 透明通道，透明背景将被压制为黑幕。")
            self.warning_label.setStyleSheet(
                "color: #FF6B6B; font-size: 12px; font-weight: bold; background: transparent; border: none;")
            self.warning_label.setWordWrap(True)
            self.warning_label.hide()
            card_layout.addWidget(self.warning_label)

        # --- 帧率选择 ---
        if self.export_type == 'video':
            self.fps_combo = OverlayComboBox()
            self.fps_combo.addItems(["24", "25", "30", "50", "60", "原视频帧率"])
            card_layout.addLayout(create_form_row("帧率", self.fps_combo))

        layout.addWidget(card_widget)
        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        cancel_btn = QPushButton("取消")
        cancel_btn.setMinimumHeight(44)
        cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        cancel_btn.setStyleSheet(
            "QPushButton { background-color: transparent; border: 1px solid #444444; color: #E0E0E0; border-radius: 8px; font-size: 15px; font-weight: bold; } QPushButton:hover { background-color: #333333; }")
        cancel_btn.clicked.connect(self.reject)

        export_btn = QPushButton("导出")
        export_btn.setMinimumHeight(44)
        export_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        export_btn.setStyleSheet(
            "QPushButton { background-color: #0A84FF; color: #FFFFFF; border-radius: 8px; font-size: 15px; font-weight: bold; border: none; } QPushButton:hover { background-color: #007AFF; } QPushButton:pressed { background-color: #0056B3; }")
        export_btn.clicked.connect(self.accept)

        btn_layout.addWidget(cancel_btn, 1)
        btn_layout.addWidget(export_btn, 1)
        layout.addLayout(btn_layout)

        main_layout.addWidget(bg_widget)

    def _init_data(self):
        desktop = os.path.join(str(Path.home()), "Desktop")
        self.path_input.setText(desktop)
        base_name, _ = os.path.splitext(self.default_name)
        self.name_input.setText(base_name)

        self.original_res_items = self.res_combo.items.copy()

        if self.export_type == 'video':
            self.fps_combo.setCurrentText("原视频帧率")
            self.format_combo.currentIndexChanged.connect(self._on_format_changed)
            self.codec_combo.currentIndexChanged.connect(self._update_warning_label)
            # 初始化触发一次联动更新
            self._on_format_changed()

        self._on_res_changed()

    def _get_codec_dict(self, fmt):
        """辅助方法：根据格式后缀返回正确的编码字典"""
        if fmt == 'mov':
            return self.CODECS_MOV
        elif fmt == 'avi':
            return self.CODECS_AVI
        return self.CODECS_MP4

    def _on_format_changed(self, index=-1):
        if self.export_type != 'video':
            return

        fmt = self.format_combo.currentText().lower()

        self.res_combo.blockSignals(True)
        if fmt == 'gif':
            self.res_combo.items = ["极速流畅小体积 (240p)", "标准清晰度 (360p)", "高清晰画质 (540p)", "超清最高限 (640p)"]
            self.res_combo.setCurrentIndex(1)
        else:
            self.res_combo.items = self.original_res_items.copy()
            self.res_combo.setCurrentIndex(0)
        self.res_combo.blockSignals(False)
        self._on_res_changed()

        self.codec_combo.blockSignals(True)
        if fmt == 'gif':
            self.codec_combo.items = ["GIF 序列帧引擎"]
            self.codec_combo.setEnabled(False)
        else:
            self.codec_combo.setEnabled(True)
            codec_list = self._get_codec_dict(fmt)
            self.codec_combo.items = list(codec_list.keys())

        self.codec_combo.setCurrentIndex(0)
        self.codec_combo.blockSignals(False)
        self._update_warning_label()

    def _update_warning_label(self, index=-1):
        if self.export_type != 'video':
            return

        fmt = self.format_combo.currentText().lower()
        if fmt == 'gif':
            self.warning_label.hide()
            return

        codec_name = self.codec_combo.currentText()
        codec_dict = self._get_codec_dict(fmt)
        current_data = codec_dict.get(codec_name, {})

        if self.has_transparent_bg and not current_data.get("alpha", False):
            self.warning_label.show()
        else:
            self.warning_label.hide()

    def _on_res_changed(self, index=-1):
        target = self.res_combo.currentText()
        orig_w, orig_h = self.original_size.width(), self.original_size.height()
        target_h = orig_h

        import re
        match = re.search(r'(\d+)[pP]', target)
        if match:
            target_h = int(match.group(1))

        if getattr(self, 'export_type', '') == 'video' and self.format_combo.currentText().lower() == 'gif':
            target_h = min(target_h, 640)

        if target_h != orig_h:
            scale = target_h / float(max(1, orig_h))
            new_w = int(orig_w * scale)
            if new_w % 2 != 0: new_w += 1
        else:
            new_w = orig_w

        self.target_size = QSize(new_w, target_h)
        self.size_info_lbl.setText(f"输出尺寸: {new_w} × {target_h}")

    def get_export_params(self):
        fmt = self.format_combo.currentText().lower()
        file_name = f"{self.name_input.text()}.{fmt}"
        full_path = os.path.join(self.path_input.text(), file_name)
        fps = self.default_fps
        codec_info = {}

        if self.export_type == 'video':
            fps_val = self.fps_combo.currentText()
            if fps_val != "原视频帧率":
                fps = float(fps_val)

            if fmt != 'gif':
                codec_name = self.codec_combo.currentText()
                codec_dict = self._get_codec_dict(fmt)
                codec_info = codec_dict.get(codec_name, {})

        return {
            'path': full_path,
            'size': self.target_size,
            'format': fmt,
            'fps': fps,
            'codec_info': codec_info
        }

    def _browse_path(self):
        directory = QFileDialog.getExistingDirectory(self, "选择导出文件夹", self.path_input.text())
        if directory:
            self.path_input.setText(directory)