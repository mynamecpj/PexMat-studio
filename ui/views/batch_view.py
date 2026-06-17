import os
import cv2

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QGridLayout, QGraphicsDropShadowEffect, QToolButton,
                               QFrame, QFileDialog, QMessageBox, QProgressDialog)
from PySide6.QtGui import QPixmap, QPainter, QColor, QCursor, QImageReader, QIcon
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QObject

from config.settings import _TR
from ui.views.image_view import ImageCompareWidget
from core.utils import imwrite_unicode, convert_cv_to_pixmap, create_checkerboard_pixmap

# --- Modern Dark Button Style System ---
BTN_PRIMARY = """
    QPushButton {
        background-color: #1A73E8; color: #FFFFFF; border-radius: 8px;
        padding: 10px 30px; font-weight: bold; font-size: 14px; border: none;
    }
    QPushButton:hover { background-color: #4C8BF5; }
    QPushButton:pressed { background-color: #145DBF; }
    QPushButton:disabled { background-color: #2A2A2A; color: #555555; }
"""

# Successful state button style
BTN_SUCCESS = """
    QPushButton {
        background-color: #059669; color: #FFFFFF; border-radius: 8px;
        padding: 10px 30px; font-weight: bold; font-size: 14px; border: none;
    }
    QPushButton:hover { background-color: #10B981; }
    QPushButton:pressed { background-color: #047857; }
"""

# Secondary button style
BTN_SECONDARY = """
    QPushButton {
        background-color: transparent; color: #E0E0E0; border-radius: 8px;
        padding: 8px 16px; font-weight: bold; font-size: 13px; border: 1px solid #404040;
    }
    QPushButton:hover { background-color: #333333; border: 1px solid #555555; }
    QPushButton:pressed { background-color: #262626; }
"""

# Danger button style
BTN_DANGER_TEXT = """
    QPushButton {
        background-color: transparent; color: #EF4444; border-radius: 6px;
        padding: 6px 12px; font-weight: bold; font-size: 12px; border: 1px solid transparent;
    }
    QPushButton:hover { background-color: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; }
"""


class BatchExportWorker(QObject):
    """Background image export worker thread to prevent freezing the UI."""
    progress = Signal(int, str)
    finished = Signal(int, int)  # success_count, total_count

    def __init__(self, export_data_list, out_dir):
        super().__init__()
        # Receives raw data (paths, cv images); do not pass QWidget instances into the worker thread.
        self.export_data_list = export_data_list
        self.out_dir = out_dir
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        count = 0
        total = len(self.export_data_list)

        for i, (file_path, bgra_cv) in enumerate(self.export_data_list):
            if self._is_cancelled:
                break

            out_name = os.path.splitext(os.path.basename(file_path))[0] + "_matted.png"
            out_path = os.path.join(self.out_dir, out_name)

            self.progress.emit(int((i / total) * 100), f"正在高速写入: {out_name}")

            if imwrite_unicode(out_path, bgra_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                count += 1

        self.progress.emit(100, "导出完成")
        self.finished.emit(count, total)


# =========================================================================
# Single Image Preview Card (Embedded side-by-side comparison, magnifier excluded)
# =========================================================================
class BatchAssetCard(QWidget):
    """Single image preview card featuring slider comparison and refinement."""
    refine_requested = Signal(object, object)
    delete_requested = Signal(object)

    def __init__(self, main_window, file_path, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.file_path = file_path
        self.mask_bool = None
        self.bgra_img_cv = None
        self.status = "待处理"

        # Retrieve actual image dimensions
        reader = QImageReader(file_path)
        sz = reader.size()
        self.resolution_text = f"{sz.width()}*{sz.height()}" if sz.isValid() else "未知尺寸"

        self.setFixedSize(200, 260)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "BatchAssetCard { background-color: #242424; border-radius: 12px; border: 1px solid #333333; }")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setBlurRadius(15)
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Upper half: slider comparison viewport
        self.compare_widget = ImageCompareWidget(self)
        self.compare_widget.setStyleSheet("background-color: #181818; border-radius: 8px;")
        self.original_pixmap = QPixmap(file_path)
        self.compare_widget.set_images(original=self.original_pixmap, enhanced=None)
        main_layout.addWidget(self.compare_widget, 1)

        # Lower half: details area
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        filename = os.path.basename(file_path)
        if len(filename) > 20: filename = filename[:8] + "..." + filename[-8:]
        self.name_label = QLabel(filename)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet(
            "color: #E0E0E0; font-size: 13px; font-weight: bold; background: transparent; border: none;")
        info_layout.addWidget(self.name_label)

        bottom_row = QHBoxLayout()
        self.res_label = QLabel(self.resolution_text)
        self.res_label.setStyleSheet("color: #666666; font-size: 11px; background: transparent; border: none;")

        # Initial mounting with formatted and translated status
        self.status_label = QLabel(_TR(f"● {self.status}"))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setStyleSheet(
            "color: #888888; font-size: 11px; font-weight: bold; background: transparent; border: none;")

        bottom_row.addWidget(self.res_label)
        bottom_row.addStretch()
        bottom_row.addWidget(self.status_label)
        info_layout.addLayout(bottom_row)

        main_layout.addLayout(info_layout, 0)

        # --- Floating button group (absolute positioning) ---
        self.overlay_widget = QWidget(self)
        self.overlay_widget.setFixedSize(80, 40)
        self.overlay_widget.move(110, 15)  # Anchor to the top-right corner
        self.overlay_widget.setStyleSheet("QWidget { background: transparent; }")

        overlay_layout = QHBoxLayout(self.overlay_widget)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(8)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        btn_style = """
            QToolButton { background-color: rgba(0,0,0,0.6); border-radius: 14px; border: 1px solid rgba(255,255,255,0.2); }
            QToolButton:hover { background-color: #1A73E8; border: 1px solid #4C8BF5; }
        """

        # Refine button (using SVG)
        self.edit_btn = QToolButton()
        self.edit_btn.setIcon(self.main_window._create_svg_icon("brush.svg", size=14, color=QColor("#FFFFFF")))
        self.edit_btn.setToolTip("精修图片")
        self.edit_btn.setFixedSize(28, 28)
        self.edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_btn.setStyleSheet(btn_style)
        self.edit_btn.setStyleSheet("""
                    QToolButton { 
                        background-color: rgba(0,0,0,0.6); 
                        color: white; 
                        border-radius: 14px; 
                        border: 1px solid rgba(255,255,255,0.2); 
                        font-size: 12px; 
                        outline: none; /* Remove the focus outline ring when clicked */
                    }
                    QToolButton:hover { 
                        background-color: rgba(255,255,255,0.2); /* Replace primary blue with highlighted semi-transparent white */
                        border: 1px solid rgba(255,255,255,0.5); 
                    }
                """)
        self.edit_btn.clicked.connect(self.ask_refine_mode)

        # Delete button (using SVG)
        self.del_btn = QToolButton()
        self.del_btn.setIcon(self.main_window._create_svg_icon("x-lg.svg", size=14, color=QColor("#FFFFFF")))
        self.del_btn.setToolTip("移除此图")
        self.del_btn.setFixedSize(28, 28)
        self.del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.del_btn.setStyleSheet("""
                    QToolButton { 
                        background-color: rgba(0,0,0,0.6); 
                        color: white; 
                        border-radius: 14px; 
                        border: 1px solid rgba(255,255,255,0.2); 
                        font-size: 12px; 
                        font-weight: bold; 
                        outline: none; /* Remove the focus outline ring when clicked */
                    }
                    QToolButton:hover { 
                        background-color: #EF4444; 
                        border: 1px solid #F87171; 
                    }
                """)
        self.del_btn.clicked.connect(lambda: self.delete_requested.emit(self))

        overlay_layout.addWidget(self.edit_btn)
        overlay_layout.addWidget(self.del_btn)

        self.overlay_widget.hide()
        self.overlay_widget.raise_()

        # Display only the delete button initially
        self.edit_btn.hide()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(10, self.compare_widget.reset_view)

    def ask_refine_mode(self):
        msg = QMessageBox(self)
        msg.setWindowTitle(_TR("精修选项"))
        msg.setText(
            _TR("请选择如何进行精修：\n\n• 继续精修：保留当前AI抠好的蒙版进行修补\n• 重新抠图：放弃当前蒙版，从原图重新开始"))
        btn_continue = msg.addButton(_TR("继续精修"), QMessageBox.ButtonRole.AcceptRole)
        btn_restart = msg.addButton(_TR("重新抠图"), QMessageBox.ButtonRole.DestructiveRole)
        msg.addButton(_TR("取消"), QMessageBox.ButtonRole.RejectRole)
        msg.setStyleSheet(
            "QMessageBox { background-color: #262626; color: #E0E0E0; } QLabel { color: #E0E0E0; } QPushButton { background-color: #333333; color: white; padding: 6px 15px; border-radius: 6px; } QPushButton:hover { background-color: #404040; }")
        msg.exec()

        if msg.clickedButton() == btn_continue:
            self.refine_requested.emit(self, self.mask_bool)
        elif msg.clickedButton() == btn_restart:
            self.refine_requested.emit(self, None)

    def update_result(self, bgra_img_cv, mask_bool, err_msg):
        if err_msg:
            self.status = "处理失败"
            self.status_label.setStyleSheet(
                "color: #EF4444; font-size: 11px; font-weight: bold; background: transparent; border: none;")
        else:
            self.status = "处理成功"
            self.status_label.setStyleSheet(
                "color: #10B981; font-size: 11px; font-weight: bold; background: transparent; border: none;")
            self.mask_bool = mask_bool
            self.bgra_img_cv = bgra_img_cv

            matted_pixmap_transparent = convert_cv_to_pixmap(bgra_img_cv)

            combined_pixmap = QPixmap(matted_pixmap_transparent.size())
            combined_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(combined_pixmap)
            painter.drawPixmap(0, 0, create_checkerboard_pixmap(combined_pixmap.size()))
            painter.drawPixmap(0, 0, matted_pixmap_transparent)
            painter.end()

            self.compare_widget.set_images(original=self.original_pixmap, enhanced=combined_pixmap)
            self.edit_btn.show()

        # Format and translate status with indicator prefix
        self.status_label.setText(_TR(f"● {self.status}"))

    def enterEvent(self, event):
        self.setStyleSheet(
            "BatchAssetCard { background-color: #2A2A2A; border-radius: 12px; border: 1px solid #1A73E8; }")
        self.overlay_widget.show()
        self.overlay_widget.raise_()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(
            "BatchAssetCard { background-color: #242424; border-radius: 12px; border: 1px solid #333333; }")
        self.overlay_widget.hide()
        super().leaveEvent(event)


# =========================================================================
# Dashed Add Image Card (AddImageCard)
# =========================================================================
class AddImageCard(QFrame):
    """Dashed add button positioned at the end of the grid."""
    add_requested = Signal()

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setFixedSize(200, 260)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QFrame { background-color: transparent; border: 2px dashed #404040; border-radius: 12px; }
            QFrame:hover { background-color: rgba(26, 115, 232, 0.05); border: 2px dashed #1A73E8; }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        icon_label = QLabel()
        # Use plus SVG icon
        icon_pixmap = self.main_window._create_svg_icon("plus-lg.svg", size=48, color=QColor("#1A73E8")).pixmap(48, 48)
        icon_label.setPixmap(icon_pixmap)
        icon_label.setStyleSheet("border: none; background: transparent;")

        text_label = QLabel("添加图片")
        text_label.setStyleSheet(
            "color: #1A73E8; font-size: 15px; font-weight: bold; border: none; background: transparent;")

        layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(text_label, 0, Qt.AlignmentFlag.AlignHCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.add_requested.emit()
        super().mousePressEvent(event)


# =========================================================================
# Batch Matting Page Container
# =========================================================================
class BatchMattingPage(QWidget):
    """Modern batch one-click matting page."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setObjectName("BatchMattingPage")
        self.setStyleSheet("QWidget#BatchMattingPage { background-color: #121212; }")

        self.cards = []
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 1. Top Status Bar ---
        self.header_widget = QWidget()
        self.header_widget.setFixedHeight(60)
        # Removed the bottom divider line for a clean dark look
        self.header_widget.setStyleSheet("background-color: #181818; border: none;")

        top_bar = QHBoxLayout(self.header_widget)
        top_bar.setContentsMargins(20, 0, 20, 0)
        top_bar.setSpacing(16)

        self.home_btn = QToolButton()
        self.home_btn.setIcon(self.main_window._create_svg_icon("house.svg", size=22, color=QColor("#E0E0E0")))
        self.home_btn.setToolTip("返回主页")
        self.home_btn.setStyleSheet(
            "QToolButton { border: none; background: transparent; border-radius: 6px; padding: 6px; } QToolButton:hover { background-color: #333333; }")
        self.home_btn.clicked.connect(
            lambda: self.main_window.switch_page(getattr(self.main_window, 'WELCOME_PAGE_INDEX', 0)))

        self.title_label = QLabel("0 张图片")
        self.title_label.setStyleSheet("color: #FFFFFF; font-size: 15px; font-weight: bold; border: none;")

        # Import folder button (SVG + uniform dark style)
        self.add_folder_btn = QPushButton("导入文件夹")
        self.add_folder_btn.setIcon(self.main_window._create_svg_icon("folder-plus.svg", size=16, color=QColor("#E0E0E0")))
        self.add_folder_btn.setStyleSheet(BTN_SECONDARY)
        self.add_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_folder_btn.clicked.connect(self.add_folder)

        # Clear list button (SVG trash can + white text + uniform dark style, no red hover)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.setIcon(self.main_window._create_svg_icon("trash.svg", size=16, color=QColor("#E0E0E0")))
        self.clear_btn.setStyleSheet(BTN_SECONDARY)
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_all_cards)

        # Layout: align both import folder and clear list buttons to the right
        top_bar.addWidget(self.home_btn)
        top_bar.addWidget(self.title_label)
        top_bar.addStretch()
        top_bar.addWidget(self.add_folder_btn)
        top_bar.addWidget(self.clear_btn)

        main_layout.addWidget(self.header_widget)

        # --- 2. Main Content Grid Area ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea { border: none; background-color: #121212; }
            QScrollBar:vertical { width: 10px; background: transparent; margin: 0px; }
            QScrollBar::handle:vertical { background: #404040; border-radius: 5px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background: #555555; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)

        self.grid_widget = QWidget()
        self.grid_widget.setStyleSheet("background-color: transparent;")
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(24, 24, 24, 24)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Initially place only one "Add Image" card
        self.add_card = AddImageCard(self.main_window)
        self.add_card.add_requested.connect(lambda: self.add_images())
        self.grid_layout.addWidget(self.add_card, 0, 0)

        self.scroll_area.setWidget(self.grid_widget)
        main_layout.addWidget(self.scroll_area, 1)

        # --- 3. Fixed Bottom Control Bar ---
        self.footer_widget = QWidget()
        self.footer_widget.setFixedHeight(70)
        # Removed the top divider line
        self.footer_widget.setStyleSheet("background-color: #181818; border: none;")

        bottom_bar = QHBoxLayout(self.footer_widget)
        bottom_bar.setContentsMargins(24, 0, 24, 0)

        self.info_label = QLabel("准备就绪")
        self.info_label.setStyleSheet("color: #888888; font-size: 13px; border: none;")

        # Export all button (SVG icon + solid white text)
        self.export_btn = QPushButton("导出全部")
        self.export_btn.setIcon(self.main_window._create_svg_icon("download.svg", size=16, color=QColor("#E0E0E0")))
        self.export_btn.setStyleSheet(BTN_SECONDARY)  # Modified to use BTN_SECONDARY
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.hide()

        # Start matting button (SVG icon + uniform dark-bordered BTN_SECONDARY, blue removed)
        self.start_btn = QPushButton("开始抠图")
        self.start_btn.setIcon(self.main_window._create_svg_icon("magic.svg", size=16, color=QColor("#E0E0E0")))
        self.start_btn.setStyleSheet(BTN_SECONDARY)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)

        bottom_bar.addWidget(self.info_label)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.export_btn)
        bottom_bar.addSpacing(10)
        bottom_bar.addWidget(self.start_btn)

        main_layout.addWidget(self.footer_widget)

    def update_ui_state(self):
        count = len(self.cards)
        # Dynamically translate "images count"
        self.title_label.setText(f"{count} {_TR('张图片')}")
        self.clear_btn.setVisible(count > 0)
        self.start_btn.setEnabled(count > 0)

        # Display the export button if all cards are successfully processed
        processed = sum(1 for c in self.cards if c.status == "处理成功" or c.status == "Success")
        if count > 0 and processed == count:
            self.export_btn.show()
            self.start_btn.hide()
            self.info_label.setText(f"{_TR('全部处理完成！(共')} {count} {_TR('张)')}")
        else:
            self.export_btn.hide()
            self.start_btn.show()
            self.info_label.setText(_TR("准备就绪") if count > 0 else _TR("请添加图片"))

    # --- Drag and Drop Logic ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            supported = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
            # Highlight the drop target if the dragged item is a directory or supported image format
            for u in urls:
                path = u.toLocalFile()
                if os.path.isdir(path) or path.lower().endswith(supported):
                    event.accept()
                    self.add_card.setStyleSheet(self.add_card.styleSheet().replace("#404040", "#1A73E8"))
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.add_card.setStyleSheet(self.add_card.styleSheet().replace("#1A73E8", "#404040"))
        event.accept()

    def dropEvent(self, event):
        self.add_card.setStyleSheet(self.add_card.styleSheet().replace("#1A73E8", "#404040"))
        urls = event.mimeData().urls()
        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')

        files_to_add = []
        for u in urls:
            path = u.toLocalFile()
            if os.path.isdir(path):
                # Traverse directory to collect supported image formats
                for root, _, files in os.walk(path):
                    for f in files:
                        if f.lower().endswith(supported):
                            files_to_add.append(os.path.join(root, f))
            elif path.lower().endswith(supported):
                files_to_add.append(path)

        if files_to_add:
            self.add_images(files_to_add)
            event.accept()
        else:
            event.ignore()

    # --- Responsive Layout ---
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rearrange_grid()

    def rearrange_grid(self):
        items = self.cards + [self.add_card]

        card_w, spacing = 200, 20
        available_w = self.scroll_area.viewport().width() - 48
        cols = max(1, available_w // (card_w + spacing))

        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            if item.widget():
                item.widget().setParent(None)

        for i, widget in enumerate(items):
            row, col = i // cols, i % cols
            self.grid_layout.addWidget(widget, row, col)

    # --- Business Logic ---
    def add_images(self, files_list=None):
        if files_list is None:
            files, _ = QFileDialog.getOpenFileNames(self, "选择多张图像", "",
                                                    "图像 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        else:
            files = files_list

        if not files: return

        existing_files = [c.file_path for c in self.cards]

        for f in files:
            if f not in existing_files:
                card = BatchAssetCard(self.main_window, f, self)
                card.refine_requested.connect(self.main_window.jump_to_refine_from_batch)
                card.delete_requested.connect(self.remove_card)
                self.cards.append(card)

        self.rearrange_grid()
        self.update_ui_state()

    def add_folder(self):
        """Import all images from a selected directory."""
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含图像的文件夹")
        if not folder_path: return

        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
        files_to_add = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(supported):
                    files_to_add.append(os.path.join(root, f))

        if not files_to_add:
            QMessageBox.information(self, "提示", "该文件夹内没有找到支持的图像文件。")
            return

        self.add_images(files_to_add)

    def remove_card(self, card):
        if card in self.cards:
            self.cards.remove(card)
            self.grid_layout.removeWidget(card)
            card.deleteLater()
            self.rearrange_grid()
            self.update_ui_state()

    def clear_all_cards(self):
        for card in self.cards:
            self.grid_layout.removeWidget(card)
            card.deleteLater()
        self.cards.clear()
        self.rearrange_grid()
        self.update_ui_state()

    def start_processing(self):
        if getattr(self.main_window, 'is_predicting', False):
            QMessageBox.warning(self, "忙碌", "其他AI任务正在进行中，请稍后再试。")
            return

        self.start_btn.setEnabled(False)
        self.info_label.setText("正在通过 AI 引擎高速处理...")
        file_paths = [c.file_path for c in self.cards]
        self.main_window._execute_batch_matting(file_paths, None, self.cards)

    def on_processing_finished(self):
        self.update_ui_state()

    def export_results(self):
        success_cards = [c for c in self.cards if c.status == "处理成功" and c.bgra_img_cv is not None]
        if not success_cards:
            QMessageBox.warning(self, "无数据", "没有可导出的抠图结果。")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "选择保存批量抠图结果的文件夹")
        if not out_dir: return

        pd = QProgressDialog("正在将透明图像写入磁盘...", "取消", 0, len(success_cards), self)

        # Hide default icon and configuration for system notices
        pd.setWindowTitle("导出进度")
        pd.setWindowIcon(QIcon())
        pd.setWindowModality(Qt.WindowModality.WindowModal)
        pd.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint
        )
        pd.setFixedSize(380, 130)

        # Inject modern dark stylesheet (the progress bar uses success green style)
        pd.setStyleSheet("""
            QProgressDialog { background-color: #1E1E1E; border: 1px solid #333333; border-radius: 8px; }
            QLabel { color: #F3F4F6; font-family: -apple-system, "Microsoft YaHei", sans-serif; font-size: 13px; margin-top: 10px; margin-bottom: 5px; }
            QProgressBar { background-color: #2D2D2D; border: none; border-radius: 4px; text-align: center; color: transparent; height: 6px; margin-bottom: 10px; }

            /* Export tasks are lighter; use a green progress bar to distinguish from core processing tasks */
            QProgressBar::chunk { background-color: #10B981; border-radius: 4px; } 

            QPushButton { background-color: #2D2D2D; color: #D1D5DB; border: none; border-radius: 6px; padding: 6px 20px; font-size: 12px; min-width: 80px; margin-top: 5px; }
            QPushButton:hover { background-color: #3F3F46; color: #FFFFFF; }
            QPushButton:pressed { background-color: #EF4444; color: #FFFFFF; }
        """)
        pd.show()

        count = 0
        for i, card in enumerate(success_cards):
            if pd.wasCanceled(): break

            out_name = os.path.splitext(os.path.basename(card.file_path))[0] + "_matted.png"
            out_path = os.path.join(out_dir, out_name)

            if imwrite_unicode(out_path, card.bgra_img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                count += 1
            pd.setValue(i + 1)

        pd.close()
        QMessageBox.information(self, "导出完毕", f"成功导出 {count} 张透明背景图像至:\n{out_dir}")