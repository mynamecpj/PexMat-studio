import traceback
from PySide6.QtWidgets import QApplication


def apply_stylesheet(app: QApplication):
    """
    [Professional Video Editing Console Pure Dark Theme]
    Force-overrides global UI styling to cultivate an immersive,
    distraction-free professional editing console environment.
    """
    C_PRIMARY = "#1A73E8"  # Highlight primary theme color (blue)
    C_PRIMARY_HOVER = "#3B82F6"
    C_PRIMARY_PRESSED = "#174EA6"
    C_PRIMARY_TEXT = "#FFFFFF"

    C_WINDOW_BG = "#181818"  # Deep dark matching the core viewport background
    C_WIDGET_BG = "#262626"  # Uniform dark gray for cards/panels
    C_INPUT_BG = "#333333"  # Standard background for inputs/lists
    C_INPUT_BG_HOVER = "#404040"
    C_INPUT_BG_DISABLED = "#202020"

    C_TITLE_CHIP_BG = "#2D3748"
    C_TITLE_CHIP_TEXT = "#90CDF4"

    C_TEXT_PRIMARY = "#E0E0E0"  # Muted off-white to reduce ocular strain
    C_TEXT_SECONDARY = "#888888"
    C_TEXT_DISABLED = "#555555"

    C_BORDER = "#333333"  # Dark border tone
    C_BORDER_FOCUS = C_PRIMARY
    C_DIVIDER_COLOR = "#2A2A2A"

    C_SECONDARY_BUTTON_BG = "transparent"
    C_SECONDARY_BUTTON_BORDER = "#444444"
    C_SECONDARY_BUTTON_TEXT = "#E0E0E0"
    C_SECONDARY_BUTTON_BG_HOVER = "#333333"

    C_LIST_ITEM_SELECTED_BG = "#1F2937"
    C_LIST_ITEM_SELECTED_TEXT = "#60A5FA"

    C_SLIDER_GROOVE_BG = "#333333"
    C_SLIDER_SUB_PAGE_BG = C_PRIMARY
    C_SLIDER_HANDLE_BG = "#E0E0E0"

    FONT_FAMILY = "'PingFang SC', 'HarmonyOS Sans SC', 'Microsoft YaHei UI', 'Segoe UI', 'Noto Sans SC', sans-serif"
    FONT_SIZE_BASE = "13px"
    FONT_SIZE_LARGE = "14px"

    C_BORDER_RADIUS_CARD = "8px"
    C_BORDER_RADIUS_BTN = "6px"
    C_BORDER_RADIUS_INPUT = "6px"

    full_style = f"""
    /* ======================================= */
    /* 1. Global Base Styles */
    /* ======================================= */
    QWidget {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_BASE};
        color: {C_TEXT_PRIMARY};
        outline: none; 
    }}
    QMainWindow {{
        background-color: {C_WINDOW_BG};
    }}
    QStatusBar {{ 
        font-size: 12px; 
        color: {C_TEXT_SECONDARY}; 
    }}

    /* ======================================= */
    /* 2. Message Box QMessageBox */
    /* ======================================= */
    QMessageBox {{
        background-color: {C_WIDGET_BG};
    }}
    QMessageBox QLabel#qt_msgbox_label, QMessageBox QLabel#qt_msgbox_informativetext {{
        color: {C_TEXT_PRIMARY};
        background-color: transparent;
    }}
    QMessageBox QPushButton {{
        background-color: {C_PRIMARY}; 
        color: {C_PRIMARY_TEXT}; 
        border: none;
        border-radius: {C_BORDER_RADIUS_BTN}; 
        padding: 6px 20px; 
        font-weight: bold; 
        min-width: 70px;
    }}
    QMessageBox QPushButton:hover {{ 
        background-color: {C_PRIMARY_HOVER}; 
    }}
    QMessageBox QPushButton:pressed {{ 
        background-color: {C_PRIMARY_PRESSED}; 
    }}

    /* ======================================= */
    /* 3. Tool Buttons and Universal Buttons */
    /* ======================================= */
    QToolButton[objectName="IconOnlyToolButton"] {{
        background-color: transparent; 
        border: none;
        border-radius: {C_BORDER_RADIUS_BTN}; 
        padding: 6px; 
        color: {C_TEXT_SECONDARY};
    }}
    QToolButton[objectName="IconOnlyToolButton"]:hover {{ 
        background-color: {C_INPUT_BG_HOVER}; 
    }}
    QToolButton[objectName="IconOnlyToolButton"]:pressed {{ 
        background-color: #555555; 
    }}
    QToolButton[objectName="IconOnlyToolButton"]:checked {{ 
        background-color: {C_TITLE_CHIP_BG}; 
        color: {C_TITLE_CHIP_TEXT};
    }}

    QToolButton#FixedHomePageButton {{
        background-color: transparent; 
        border: none;
        border-radius: 18px; 
    }}

    QToolButton#TopPanelButton {{
        background-color: transparent; 
        color: {C_TEXT_SECONDARY};
        border: none; 
        border-radius: {C_BORDER_RADIUS_BTN};
        padding: 8px 16px; 
        font-weight: 500; 
    }}
    QToolButton#TopPanelButton:hover {{ 
        background-color: {C_INPUT_BG_HOVER}; 
    }}
    QToolButton#TopPanelButton:checked {{ 
        background-color: {C_TITLE_CHIP_BG}; 
        color: {C_TITLE_CHIP_TEXT}; 
        font-weight: 600; 
    }}

    QPushButton, QPushButton#PrimaryButton {{
        background-color: {C_PRIMARY}; 
        color: {C_PRIMARY_TEXT}; 
        border: none;
        border-radius: {C_BORDER_RADIUS_BTN}; 
        padding: 8px 20px;
        font-weight: bold; 
        font-size: {FONT_SIZE_LARGE};
    }}
    QPushButton:hover, QPushButton#PrimaryButton:hover {{ 
        background-color: {C_PRIMARY_HOVER}; 
    }}
    QPushButton:pressed, QPushButton#PrimaryButton:pressed {{ 
        background-color: {C_PRIMARY_PRESSED}; 
    }}
    QPushButton:disabled, QPushButton#PrimaryButton:disabled {{ 
        background-color: {C_INPUT_BG_DISABLED}; 
        color: {C_TEXT_DISABLED}; 
    }}

    QPushButton#SecondaryButton {{
        background-color: {C_SECONDARY_BUTTON_BG}; 
        color: {C_SECONDARY_BUTTON_TEXT}; 
        font-weight: bold;
        border: 1px solid {C_SECONDARY_BUTTON_BORDER};
    }}
    QPushButton#SecondaryButton:hover {{ 
        background-color: {C_SECONDARY_BUTTON_BG_HOVER}; 
        border: 1px solid {C_BORDER};
    }}
    QPushButton#SecondaryButton:pressed {{ 
        background-color: #222222; 
    }}

    /* ======================================= */
    /* 4. Panels and Card Containers */
    /* ======================================= */
    SlidingPanelFrame, QWidget#CardWidget {{
        background-color: {C_WIDGET_BG}; 
        border: 1px solid {C_BORDER}; 
        border-radius: {C_BORDER_RADIUS_CARD};
    }}
    QFrame#FloatingPanelContentFrame, AssetPanelFrame {{ 
        background-color: {C_WIDGET_BG}; 
        border: 1px solid {C_BORDER}; 
        border-radius: {C_BORDER_RADIUS_CARD}; 
    }}
    QLabel#CardTitleLabel {{
        background-color: transparent; 
        color: {C_TEXT_PRIMARY};
        padding: 4px 0px; 
        border: none;
        font-size: {FONT_SIZE_LARGE}; 
        font-weight: bold; 
        margin-bottom: 4px;
    }}

    /* ======================================= */
    /* 5. Input Widgets */
    /* ======================================= */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
        background-color: {C_INPUT_BG}; 
        color: {C_TEXT_PRIMARY};
        border: 1px solid {C_BORDER}; 
        border-radius: {C_BORDER_RADIUS_INPUT}; 
        padding: 6px 12px; 
        min-height: 24px;
        selection-background-color: {C_PRIMARY};
        selection-color: {C_PRIMARY_TEXT};
    }}
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QPlainTextEdit:hover {{
        background-color: {C_INPUT_BG_HOVER};
        border: 1px solid #444444;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
        background-color: #1A1A1A;
        border: 1px solid {C_BORDER_FOCUS};
    }}

    QSpinBox, QDoubleSpinBox {{ padding-right: 0px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: transparent;
        border: none; 
        width: 24px;
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: #444444; 
        border-radius: 4px;
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        width: 0; height: 0; 
        border-left: 4px solid transparent;
        border-right: 4px solid transparent; 
        border-bottom: 5px solid {C_TEXT_SECONDARY};
    }}
    QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{ border-bottom-color: {C_TEXT_PRIMARY}; }}

    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        width: 0; height: 0; 
        border-left: 4px solid transparent;
        border-right: 4px solid transparent; 
        border-top: 5px solid {C_TEXT_SECONDARY};
    }}
    QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{ border-top-color: {C_TEXT_PRIMARY}; }}

    QComboBox {{ padding-right: 12px; }}
    QComboBox::drop-down {{ 
        border: none; 
        background: transparent; 
        width: 0px; 
    }}
    QComboBox::down-arrow {{ image: none; }}

    QComboBox QAbstractItemView {{
        border: 1px solid {C_BORDER}; 
        border-radius: {C_BORDER_RADIUS_INPUT}; 
        background-color: {C_INPUT_BG};
        selection-background-color: {C_LIST_ITEM_SELECTED_BG}; 
        selection-color: {C_LIST_ITEM_SELECTED_TEXT};
        padding: 4px; 
        outline: none;
    }}
    QComboBox QAbstractItemView::item {{ 
        padding: 8px 12px; 
        border-radius: 4px; 
        color: {C_TEXT_PRIMARY};
    }}

    /* ======================================= */
    /* 6. List Widget (QListWidget) */
    /* ======================================= */
    QListWidget {{
        border-radius: {C_BORDER_RADIUS_CARD}; 
        padding: 4px;
        border: 1px solid {C_BORDER}; 
        background-color: {C_INPUT_BG}; 
        outline: none;
    }}
    QListWidget::item {{ 
        padding: 8px 12px; 
        border-radius: 6px; 
        color: {C_TEXT_PRIMARY};
    }}
    QListWidget::item:hover {{ 
        background-color: {C_INPUT_BG_HOVER}; 
    }}
    QListWidget::item:selected {{ 
        background-color: {C_LIST_ITEM_SELECTED_BG}; 
        color: {C_LIST_ITEM_SELECTED_TEXT}; 
        font-weight: bold; 
    }}

    /* ======================================= */
    /* 7. Slider Styles */
    /* ======================================= */
    QSlider[orientation="1"] {{ 
        padding: 10px 0px; 
    }}
    QSlider::groove:horizontal {{
        border: none; 
        height: 4px; 
        border-radius: 2px;
        background: {C_SLIDER_GROOVE_BG};
    }}
    QSlider::sub-page:horizontal {{
        background: {C_SLIDER_SUB_PAGE_BG}; 
        height: 4px; 
        border-radius: 2px;
    }}
    QSlider::add-page:horizontal {{ 
        background: transparent; 
    }}
    QSlider::handle:horizontal {{
        background: {C_SLIDER_HANDLE_BG};
        border: none;
        width: 14px; 
        height: 14px;
        margin: -5px 0px; 
        border-radius: 7px; 
    }}
    QSlider::handle:horizontal:hover {{ 
        width: 16px; 
        height: 16px;
        margin: -6px -1px;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:pressed {{ 
        background: {C_PRIMARY_HOVER}; 
    }}

    /* ======================================= */
    /* 8. Miscellaneous Components */
    /* ======================================= */
    QFrame {{ 
        border: none; 
        background-color: transparent; 
    }}
    QCheckBox, QRadioButton {{
        spacing: 8px;
        color: {C_TEXT_PRIMARY};
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid {C_TEXT_SECONDARY};
        background-color: transparent;
    }}
    QCheckBox::indicator:hover {{
        border-color: {C_PRIMARY};
    }}
    QCheckBox::indicator:checked {{
        background-color: {C_PRIMARY};
        border: 2px solid {C_PRIMARY};
        image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/%3E%3C/svg%3E");
    }}
    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 9px; 
        border: 2px solid {C_TEXT_SECONDARY};
        background-color: transparent;
    }}
    QRadioButton::indicator:hover {{
        border-color: {C_PRIMARY};
    }}
    QRadioButton::indicator:checked {{
        border: 2px solid {C_PRIMARY};
        background-color: transparent;
        image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='5' fill='%231A73E8'/%3E%3C/svg%3E");
    }}

    /* Vertical Scrollbar Structure */
    QScrollBar:vertical {{
        background-color: transparent;
        width: 10px;
        margin: 0px 2px 0px 2px; /* Fine margin alignment */
    }}

    /* Horizontal Scrollbar Structure */
    QScrollBar:horizontal {{
        background-color: transparent;
        height: 10px;
        margin: 2px 0px 2px 0px;
    }}

    /* Scrollbar Handle Body */
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background-color: #4A4A4A; /* Muted dark gray */
        border-radius: 3px;        /* Smooth curvature */
        min-height: 30px;
        min-width: 30px;
    }}

    /* Highlight handle on mouse hover */
    QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
        background-color: #6B7280;
    }}

    /* Apply primary color on mouse press */
    QScrollBar::handle:vertical:pressed, QScrollBar::handle:horizontal:pressed {{
        background-color: #1A73E8;
    }}

    /* Hide adjustment arrow buttons (deprecated in modern UI style guidelines) */
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        height: 0px; 
        width: 0px;
        border: none;
        background: transparent;
    }}

    /* Set track background to transparent above/below and left/right of handle */
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: transparent;
    }}

    /* Strip borders from scroll areas to blend seamlessly with the backdrop */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    """
    try:
        app.setStyleSheet(full_style)
        print("已应用专业的沉浸式纯黑视频剪辑台 UI 样式表。")
    except Exception as e:
        print(f"应用样式表时出错: {e}")