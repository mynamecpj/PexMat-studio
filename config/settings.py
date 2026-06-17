import collections
from PySide6.QtGui import QColor

# ==============================================================================
# --- Configuration (Model Paths & Global Variables) ---
# ==============================================================================

# Segment Anything Model 2 (SAM2) Configuration
SAM2_IMAGE_CHECKPOINT_PATH = "checkpoints/sam2.1_hiera_large.pt"
SAM2_IMAGE_MODEL_CFG_PATH = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_VIDEO_CHECKPOINT_PATH = "checkpoints/sam2.1_hiera_small.pt"
SAM2_VIDEO_MODEL_CFG_PATH = "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"

# ==============================================================================
# --- MatAnyone 2 Configuration ---
# ==============================================================================
MATANYONE_CHECKPOINT_PATH = "checkpoints/matanyone2.pth"

# Real-ESRGAN Upscaling and Enhancement Models
ENHANCE_MODELS = {
    "动漫": "checkpoints/RealESRGAN_x4plus_anime_6B.pth",
    "通用": "checkpoints/RealESRGAN_x4plus.pth",
}
DEFAULT_ENHANCE_MODEL_NAME = "动漫"
ENHANCE_FIXED_DENOISE = 0.5

# Workspace Limitations & Viewport Settings
MAX_UNDO_HISTORY = 20
TEMP_BASE_DIR = "temp_toolbox_session"
MIN_ZOOM = 0.05
MAX_ZOOM = 50.0
ZOOM_FACTOR = 1.15

# ==============================================================================
# --- Canvas Background Checkerboard Colors (Premium Dark Theme) ---
# ==============================================================================
PREVIEW_BG_CHECKER_SIZE = 32
PREVIEW_BG_COLOR1 = QColor("#141414")  # Ultra-dark gray
PREVIEW_BG_COLOR2 = QColor("#1C1C1C")  # Slightly lighter charcoal gray

# Mask Visualization Configurations
DEFAULT_MASK_ALPHA_IMAGE = int(0.5 * 255)
MASK_COLORS = collections.OrderedDict([
    ("蓝色", QColor(60, 120, 220)), ("红色", QColor(230, 50, 50)), ("绿色", QColor(50, 200, 50)),
    ("黄色", QColor(255, 210, 0)), ("品红", QColor(230, 0, 230)), ("青色", QColor(0, 200, 200)),
    ("橙色", QColor(255, 128, 0)), ("紫色", QColor(128, 0, 128)), ("灰色", QColor(128, 128, 128)),
])
DEFAULT_MASK_COLOR_NAME = "蓝色"

# Mask Edge Refinement Parameters
DEFAULT_REFINE_SMOOTH = 0
MAX_REFINE_SMOOTH = 25
DEFAULT_REFINE_FEATHER = 0
MAX_REFINE_FEATHER = 50
DEFAULT_REFINE_SHIFT = 0
MAX_REFINE_SHIFT_INTERNAL = 25
MAX_REFINE_SHIFT_SLIDER = MAX_REFINE_SHIFT_INTERNAL * 5
SHIFT_SLIDER_FACTOR = 5.0
DEFAULT_REFINE_SHIFT_SLIDER = DEFAULT_REFINE_SHIFT * SHIFT_SLIDER_FACTOR
DEFAULT_REFINE_GUIDED_FILTER_ENABLED = False
DEFAULT_REFINE_GUIDED_FILTER_RADIUS = 5
DEFAULT_REFINE_GUIDED_FILTER_EPS_SCALED = 100
MAX_REFINE_GUIDED_FILTER_RADIUS = 50
MAX_REFINE_GUIDED_FILTER_EPS_SCALED = 1000
REFINEMENT_UPDATE_DEBOUNCE_MS = 30

# UI Window & Panel Sizing Layout Constraints
CONTROLS_FIXED_WIDTH_IMG_SEG = 380
CONTROLS_FIXED_WIDTH_VID_SEG = 400
WINDOW_MIN_WIDTH = 1280
WINDOW_MIN_HEIGHT = 720
APP_ICON_FILENAME = "app_icon.ico"

# Video Processing & Playback Configurations
VIDEO_PLAYBACK_INTERVAL_MS = 40
VIDEO_DEFAULT_FPS = 25.0
FRAME_EXTRACT_EVERY_N = 1
VIDEO_SAVE_FPS = 25.0
VIDEO_SAVE_CODEC_MP4 = 'mp4v'
VIDEO_SAVE_CODEC_AVI = 'XVID'
VIDEO_SAVE_CODEC_MOV = 'mp4v'
DEFAULT_VIDEO_BG_COLOR = QColor(0, 255, 0)
VIDEO_TARGET_COLORS = [
    QColor(60, 120, 220), QColor(230, 50, 50), QColor(50, 200, 50),
    QColor(255, 210, 0), QColor(230, 0, 230), QColor(0, 200, 200),
    QColor(255, 128, 0), QColor(128, 0, 128), QColor(128, 128, 128),
    QColor(255, 192, 203), QColor(0, 128, 0), QColor(100, 100, 255)
]
VIDEO_OBJ_COLORS = VIDEO_TARGET_COLORS
MAX_VIDEO_OBJS = len(VIDEO_TARGET_COLORS)
VIDEO_POINT_RADIUS_IMG = 4
DEFAULT_MASK_ALPHA_VIDEO = int(0.5 * 255)
VIDEO_CLICK_PREDICT_MASK_ALPHA = DEFAULT_MASK_ALPHA_VIDEO
VIDEO_FRAME_EXT = ".jpg"
VIDEO_THUMBNAIL_EXT = ".jpg"
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.gif')
GIF_DEFAULT_DURATION_MS = 100
GIF_SAVE_OPTIMIZE = True

# Worker Thread Mappings for Logging and Status Updates
WORKER_ID_TO_CN = {
    "load_image_model": "加载图像模型",
    "load_video_model": "加载视频模型",
    "load_matanyone_model": "加载精细视频模型",
    "enhance": "增强",
    "predict": "预测",
    "extract": "提取帧",
    "propagate_video_v1991": "视频传播",
    "save_enhanced": "保存增强结果",
    "save_segment": "保存分割结果",
    "save_video": "保存视频",
    "save_stitched": "保存拼接结果"
}

# Image Dimension & Canvas Presets
MAX_WORKING_DIM_SEGMENTATION = 1280
STITCHING_DEFAULT_CANVAS_WIDTH = 1920
STITCHING_DEFAULT_CANVAS_HEIGHT = 1080
STITCHING_HANDLE_SIZE = 5

# ==============================================================================
# --- UI Color Palette (Premium Dark Mode Edition) ---
# ==============================================================================
C_PRIMARY = "#1A73E8"
C_PRIMARY_HOVER = "#3B82F6"
C_PRIMARY_PRESSED = "#174EA6"
C_PRIMARY_TEXT = "#FFFFFF"

C_WINDOW_BG = "#181818"          # Deep main window background
C_WIDGET_BG = "#262626"          # Cards and container widgets
C_INPUT_BG = "#333333"           # Textboxes and standard input fields
C_INPUT_BG_DISABLED = "#202020"  # Disabled background styling
C_TITLE_CHIP_BG = "#2D3748"      # Subheader labels / chips

C_TEXT_PRIMARY = "#E0E0E0"       # High-contrast readable gray
C_TEXT_SECONDARY = "#888888"     # Subtitle and secondary text
C_TEXT_DISABLED = "#555555"      # Disabled state typography

C_BORDER = "#333333"             # Standard subtle borders
C_BORDER_INPUT_HOVER = "#444444"
C_BORDER_INPUT_FOCUS = C_PRIMARY
C_DIVIDER_COLOR = "#2A2A2A"      # Divider line accent

C_SECONDARY_BUTTON_BG = "#333333"
C_SECONDARY_BUTTON_BG_HOVER = "#404040"
C_SECONDARY_BUTTON_TEXT = "#E0E0E0"

C_LIST_ITEM_SELECTED_BG = "#1F2937"
C_LIST_ITEM_SELECTED_TEXT = "#60A5FA"

C_SLIDER_GROOVE_BG = "#333333"
C_SLIDER_SUB_PAGE_BG = C_PRIMARY
C_SLIDER_HANDLE_BG = "#E0E0E0"
C_SLIDER_HANDLE_BORDER = "#333333"
C_SLIDER_HANDLE_BORDER_HOVER = C_PRIMARY_HOVER

C_SCROLLBAR_BG = "transparent"
C_SCROLLBAR_HANDLE = "#4A4A4A"
C_SCROLLBAR_HANDLE_HOVER = "#6B7280"

C_WELCOME_CARD_BG = C_WIDGET_BG
C_WELCOME_CARD_TITLE_TEXT = C_TEXT_PRIMARY

FONT_FAMILY = "Microsoft YaHei, Segoe UI, SimHei, sans-serif"
FONT_SIZE_BASE = "9.5pt"
FONT_SIZE_LARGE = "10.5pt"
C_BORDER_RADIUS = "16px"
C_BORDER_RADIUS_SM = "10px"

C_SECONDARY_TEXT = "#888888"
C_ICON_COLOR_TOOL = QColor("#E0E0E0") # Vector icons unified color scheme

# ==============================================================================
# --- Global Internationalization (I18N) Engine ---
# ==============================================================================
APP_STATE = {
    "LANG": "zh"
}

I18N_DICT = {
    # --- Welcome & Batch Pages ---
    "创意": "Creativity",
    " 从这里开始": " Begin",
    "打开文件": "Open File",
    "快速高清": "Quick Enhance",
    "快速抠图": "Quick Cutout",
    "批量抠图": "Batch Matting",
    "剪贴板创建": "From Clipboard",
    "最近打开": "Recent Files",
    "暂无历史记录": "No Recent Files",
    "删除此记录": "Delete Record",
    "GPU 加速": "GPU Accel",
    "GPU 加速 (无硬件)": "GPU Accel (N/A)",
    "虚化动画": "Blur Anim",
    "导入文件夹": "Import Folder",
    "清空列表": "Clear List",
    "导出全部": "Export All",
    "开始抠图": "Start Matting",
    "准备就绪": "Ready",
    "添加图片": "Add Image",
    "待处理": "Pending",
    "处理成功": "Success",
    "处理失败": "Failed",
    "佩克斯的魔法屋": "Pex's Magic House",
    "选择颜色": "Select Color",
    "当前色": "Current Color",

    # --- Top Bar & General Dialogs ---
    "主页": "Home",
    "导出": "Export",
    "取消": "Cancel",
    "确定": "Confirm",
    "完成": "Done",
    "处理中": "Processing",
    "系统处理中": "System Processing",
    "系统加载中": "System Loading",

    # --- Creative Workshop (Canvas & Layers) ---
    "创意工坊": "Creative Workshop",
    "素材库": "Asset Library",
    "操作": "Actions",
    "画布": "Canvas",
    "图层": "Layers",
    "画布设置": "Canvas Settings",
    "宽度:": "Width:",
    "高度:": "Height:",
    "透明背景": "Transparent BG",
    "纯色": "Solid Color",
    "选中素材属性": "Item Properties",
    "X:": "X:",
    "Y:": "Y:",
    "宽:": "W:",
    "高:": "H:",
    "旋转:": "Rotate:",
    "合并选中项": "Merge Selected",
    "取消合并": "Uncombine",
    "图层与顺序": "Layers & Order",
    "添加新素材": "Add New Asset",
    "图层列表": "Layer List",
    "选择拼图画布尺寸预设": "Select Canvas Size Preset",
    "画布背景颜色": "Canvas Background Color",
    "选择画布背景颜色": "Select Canvas Background Color",
    "图层 ": "Layer ",
    "保存拼接图像": "Save Stitching Image",
    "画布上没有任何内容。": "Canvas is completely empty.",
    "正在准备保存...": "Preparing to save...",
    "选择要拼接的图像 (可多选)": "Select Images to Stitch (Multi-select)",
    "画布上已经没有内容了。": "Canvas has no contents remaining.",
    "清空画布": "Clear Canvas",
    "您确定要移除画布上的所有图像吗？\n此操作无法撤销。": "Are you sure you want to remove all images from the canvas?\nThis action cannot be undone.",
    "无法加载素材：\n": "Cannot load asset:\n",

    # --- Creative Workshop (Segmentation & Refining) ---
    "一键智能主体抠图": "1-Click Auto Matting",
    "智能框选模式 (SAM2)": "Smart Box (SAM2)",
    "在当前蒙版上累加 (Shift)": "Add to Current Mask (Shift)",
    "手工画笔模式 (Brush)": "Manual Brush Mode",
    "粗细:": "Size:",
    "流畅笔迹实时预览": "Live Brush Preview",
    "蒙版边缘优化": "Mask Edge Refinement",
    "启用极致发丝抠图 (耗时较长)": "Enable Fine Hair Matting (Slow)",
    "边缘平滑:": "Edge Smooth:",
    "边缘羽化:": "Edge Feather:",
    "收缩/扩张:": "Shrink/Expand:",
    "蒙版颜色": "Mask Color",
    "完成并返回": "Done & Return",
    "撤销": "Undo",
    "重做": "Redo",
    "保存": "Save",
    "点 / 框": "Point / Box",
    "画笔修补": "Brush Patch",
    "点 / 框选抠图": "Point / Box Select",
    "开始自动抠图": "Start Auto Matting",
    "一键自动抠图 (智能大模型引擎)": "1-Click Auto Matting (Smart Large Model Engine)",
    "正在一键抠图 (": "Running 1-Click Matting (",
    "正在启动一键自动抠图 (智能大模型引擎)...": "Starting 1-click auto matting (Smart Large Model Engine)...",
    "正在启动一键自动抠图| 推理设备: ": "Starting 1-click auto matting | Device: ",
    "滑块延迟": "Slider Delay",
    "画笔大小": "Brush Size",
    "选择画笔颜色": "Select Brush Color",
    "蒙版调整": "Mask Adjustments",
    "粗细": "Brush Size",
    "缩放": "Scale",
    "腐蚀": "Erode",
    "膨胀": "Dilate",
    "激活发丝精雕引擎": "Activate Hair Refining Engine",
    "无需定位直达发丝精抠": "Full-frame Direct Hair Matting",
    "删除": "Delete",

    # --- Resolution Selector Panels ---
    "工作分辨率": "Working Resolution",
    "原图尺寸": "Original Size",
    "512像素 (推荐CPU)": "512px (Rec. CPU)",
    "768像素": "768px",
    "1280像素 (推荐GPU)": "1280px (Rec. GPU)",
    "1920像素": "1920px",
    "自定义尺寸": "Custom Size",
    "最大尺寸:": "Max Size:",

    # --- Upscaling & HD Enhancement Panels ---
    "放大参数设置": "Upscale Settings",
    "放大倍率:": "Upscale Ratio:",
    "分块模式:": "Tile Mode:",
    "选择模型:": "Select Model:",
    "自动 (推荐)": "Auto (Rec)",
    "动漫": "Anime",
    "通用": "General",
    "开始增强": "Start Enhance",
    "保存结果": "Save Result",
    "增强效果预览": "Enhance Preview",
    "大图模式": "Large Image Mode",
    "中图模式": "Medium Image Mode",
    "小图模式": "Small Image Mode",
    "分块模式": "Tile Mode",
    "自动根据图像分辨率选择合适的分块大小，平衡速度与显存。": "Auto-select tile size based on resolution to balance speed and VRAM.",
    "手动指定分块大小（越小越省显存，但处理更慢且可能有接缝）。": "Manually specify tile size (smaller saves VRAM but slower and may have seams).",
    "强制使用 128 分块，适合极高分辨率（极低显存占用）。": "Force 128 tiling, best for extreme resolutions (extremely low VRAM).",
    "强制使用 256 分块，适合 2K-4K 图像（较低显存占用）。": "Force 256 tiling, best for 2K-4K images (lower VRAM).",
    "强制使用 512 分块，适合 1080P 以下（较高显存占用）。": "Force 512 tiling, best for under 1080P (higher VRAM).",
    "选择分块处理模式以平衡显存占用和处理速度。": "Choose tile processing mode to balance VRAM usage and processing speed.",
    "选择一张图片进行高清增强": "Select an Image for HD Upscaling",
    "倍率获取": "Ratio Retrieval",
    "选择的增强模型无效。": "The selected upscaling model is invalid.",
    "增强模型未找到:": "Upscaling model not found:",
    "增强中 ": "Upscaling ",
    "增强素材失败:": "Failed to upscale asset:",
    "增强倍数不能小于1x，已自动设为1x。": "Upscaling ratio cannot be less than 1x, reset to 1x automatically.",
    "没有可保存的增强结果。请先应用增强。": "No enhanced result to save. Please run upscaling first.",
    "增强图像": "Upscale Image",
    "保存增强图像 (PNG)": "Save Upscaled Image (PNG)",

    # --- Batch Matting Controls ---
    "选择多张图像进行批量一键抠图": "Select Multiple Images for Batch Matting",
    "选择保存批量抠图结果的文件夹": "Select Output Folder for Batch Matting",
    "准备开始批量处理 ": "Preparing to process batch of ",
    "正在高速处理图像...": "Processing images at high speed...",
    "中止并保存": "Abort & Save",
    "批量处理进度": "Batch Progress",
    "批量抠图已中止，已成功处理 ": "Batch matting aborted, successfully processed ",
    "批量抠图完成": "Batch Matting Complete",
    "批量处理结束！\n成功生成 ": "Batch processing finished!\nSuccessfully generated ",
    " 张透明图像。": " transparent images.",
    "批量抠图成功处理 ": "Successfully processed batch of ",
    "批量抠图失败": "Batch Matting Failed",
    "处理过程中断:\n": "Processing interrupted:\n",
    "\n已成功输出 ": "\nSuccessfully exported ",
    "正在高速提取主图像素...": "Extracting main pixels at high speed...",
    "正在高速写入: ": "Writing to disk at high speed: ",
    "已成功抠出 ": "Successfully matted ",
    " 张图像！\n您可以点击卡片进行对比或精修，确认无误后点击顶部【导出全部】保存到电脑。": " images!\nYou can click cards to compare or refine, then click 'Export All' to save to your PC.",
    "批量抠图发生错误。": "An error occurred during batch matting.",
    "正在将透明图像写入磁盘...": "Writing transparent images to disk...",
    "导出完毕": "Export Complete",
    "成功导出 ": "Successfully exported ",
    " 张透明背景图像至:\n": " transparent background images to:\n",
    "● 待处理": "● Pending",
    "● 处理成功": "● Success",
    "● 处理失败": "● Failed",

    # --- Video Editor Panel ---
    "视频抠图": "Video Matting",
    "图层": "Layer",
    "张图片": "Images",
    "对象": "Target",
    "目标": "Target",
    "全部处理完成！(共": "All processing complete! (Total ",
    "张)": " images)",
    "请添加图片": "Please add images",
    "标注于帧": "Annotated on Frame",
    "点数:": "Points:",
    "项目库": "Project Library",
    "你的项目库为空": "Your Library is Empty",
    "+ 添加": "+ Add",
    "合并栏": "Combine Clips",
    "剪裁": "Crop",
    "进入剪辑": "Crop",
    "智能抠图": "Auto Matting",
    "拖拽到此添加并合并视频": "Drag & Drop to Add and Merge Videos",
    "剪辑长度": "Duration",
    "拖放视频/GIF文件至此，\n或使用“加载”按钮，\n支持缩放/平移。": "Drag & drop Video/GIF here,\nor use 'Load' button,\nsupports zoom/pan.",
    "视频未加载": "Video Not Loaded",
    "追踪对象": "Tracked Objects",
    "+ 添加新对象": "+ Add Target",
    "删除对象": "Delete Target",
    "开始智能渲染": "Start Smart Render",
    "恢复原声": "Restore Audio",
    "关闭原声": "Mute Audio",
    "原音量": "Orig Vol",
    "音乐": "Music",
    "音乐剪切区间:": "Music Trim Interval:",
    "视频加入时刻:": "Insert at:",
    "乐音量": "BGM Vol",
    "视频背景颜色": "Video Background Color",
    "选择视频背景颜色": "Select Video Background Color",
    "当前视频保存背景颜色: ": "Current save background color: ",
    "点击更改": "Click to Change",
    "静音当前片段": "Mute Current Clip",
    "剪辑与音频": "Clip & Audio",
    "剪辑时间信息": "Clip Time Info",
    "原声音轨": "Original Audio",
    "背景": "Background",
    "透明背景 (GIF合并支持)": "Transparent BG (GIF support)",
    "选择纯色背景": "Select Solid Color Background",
    "选择自定义背景": "Select Custom Background",
    "无 (请先添加)": "None (Add First)",
    "无背景音乐": "No Background Music",
    "视频抠图结果预览": "Video Matting Preview",
    "视频流提取失败": "Video Extraction Failed",
    "无法完成物理帧同步，请检查视频文件路径是否正常。\n\n错误信息: ": "Failed to perform physical frame sync. Verify video path.\n\nError: ",
    "正在唤醒 AI 引擎并重构历史图层，请勿操作...": "Waking up AI engine & reconstructing layers, please do not interact...",
    "正在唤醒 AI 视频引擎...": "Waking up AI video engine...",
    "视频环境准备完毕，操作已解锁！": "Video environment ready, interaction unlocked!",
    "视频特征编码完毕。": "Video feature encoding complete.",
    "初始化视频预测器状态失败:": "Failed to initialize video predictor state:",
    "正在重置硬件显存与底层模型，请稍候...": "Resetting hardware VRAM & underlying models, please wait...",
    "正在初始化智能视频抠图传播引擎...": "Initializing video matting propagation engine...",
    "开始时序追踪...": "Starting temporal tracking...",
    "正在准备视频特征追踪流...": "Preparing video feature tracking streams...",
    "正在启动图层定位并同步追踪序列...": "Initializing layers & synchronizing tracking sequences...",
    "智能画面解析中: ": "Parsing frames intelligently: ",
    "视频追踪处理完毕。": "Video tracking complete.",
    "视频分割处理已取消。": "Video segmentation cancelled.",
    "视频分割处理出错: ": "Video segmentation error: ",
    "正向发丝追踪:": "Forward Hair Tracking:",
    "反向发丝追踪:": "Backward Hair Tracking:",
    "正向发丝提取:": "Forward Hair Extraction:",
    "反向发丝提取:": "Backward Hair Extraction:",
    "正在合并双向稳定时序流...": "Merging bidirectional stable temporal flows...",
    "时序追踪优化解算完成！": "Temporal tracking optimization complete!",
    "发丝动态时序跟踪传播异常: ": "Hair tracking temporal propagation anomaly: ",
    "粗定位运动追踪:": "Coarse Trajectory Tracking:",
    "时序精抠": "Temporal Matting",
    "视频已从活动列表安全移除。": "Video successfully removed from active workers.",
    "状态: 抠图完成": "Status: Matting Complete",
    "状态: 准备抠图": "Status: Ready",
    "状态: 已加载": "Status: Loaded",
    "状态: 未加载素材": "Status: No Asset Loaded",
    "状态: 取消中...": "Status: Cancelling...",
    "状态: 已取消": "Status: Cancelled",
    "正在混合多音频轨并启动视频编码器...": "Blending multiple audio tracks & initiating video encoder...",
    "正在混合多音频轨并保存视频...": "Blending multiple audio tracks & saving video...",
    "正在合并时序帧序列...": "Merging temporal frame sequences...",
    "合并帧序列完成": "Frame sequence merge complete",
    "正在混合音视频流并导出...": "Merging audio-video streams and exporting...",
    "正在无损重构视频片段...\n": "Losslessly reconstructing video clips...\n",
    "发丝级双引擎优化渲染中...\n": "Dual-engine hair-level optimization rendering...\n",
    "正在提取视频特征深度编码...\n": "Extracting deep video feature encodings...\n",
    "正在混合多轨音视频流并导出...\n": "Merging multi-track audio-video streams & exporting...\n",
    "系统处理中，请稍候...\n": "System processing, please wait...\n",

    # --- Framework Core & Underlying AI Pipeline States ---
    "正在将历史图层注入 AI 引擎...\n这可能需要 1~2 秒，请稍候。": "Injecting historical layers to AI...\nThis may take 1-2s, please wait.",
    "正在编码视频特征，首次读取或切换硬件可能需要数秒...": "Encoding video features, first load or HW switch may take seconds...",
    "视频编码完成！\n正在注入 AI 引擎与重建图层...": "Encoding done!\nInjecting AI engine & rebuilding layers...",
    "正在分割目标，请稍候...": "Segmenting target, please wait...",
    "正在抽取并构建剪辑沙盒，请稍候...": "Building clip sandbox, please wait...",
    "正在计算图层结构": "Calculating layer structure...",
    "正在进行精细优化...": "Running fine optimization...",
    "正在唤醒 AI 视频引擎...": "Waking up AI video engine...",
    "准备图像以用于分割预测器...": "Preparing image for predictor...",
    "正在在后台线程中执行 SAM 2 图像特征编码（set_image）...": "Encoding SAM 2 image features (set_image) in background thread...",
    "正在重置硬件显存与底层模型，请稍候。": "Resetting hardware memory and models, please wait.",
    "正在初始化发丝级大模型，请稍候...": "Initializing fine hair model, please wait...",
    "优化参数已改变，准备全局重绘: ": "Refinement parameters changed, preparing full redraw: ",
    "收到新增抠图区域，准备局部精修...": "New matted region received, preparing local refinement...",
    "正在精细优化...": "Optimizing details...",
    "正在重载大图...": "Reloading high-res image...",
    "正在加载图片...": "Loading image...",
    "正在重构图层对象 ": "Reconstructing layer object ",

    # --- Overlay Windows, Headers & Prompt Boxes ---
    "提示": "Notice",
    "错误": "Error",
    "警告": "Warning",
    "加载错误": "Load Error",
    "保存成功": "Save Success",
    "保存失败": "Save Failed",
    "保存确认": "Save Confirmation",
    "确认删除": "Confirm Delete",
    "确认清除": "Confirm Clear",
    "无需清除": "No Need to Clear",
    "确认重置抠图状态": "Confirm Reset",
    "确认加载新图": "Confirm Load New",
    "确认更改分辨率": "Confirm Resolution",
    "确认开始": "Confirm Start",
    "操作提示": "Operation Tip",
    "帧不匹配": "Frame Mismatch",
    "处理提示": "Process Tip",
    "操作失败": "Operation Failed",
    "无法添加": "Cannot Add",
    "无法删除": "Cannot Delete",
    "无法保存": "Cannot Save",
    "忙碌": "Busy",
    "需要操作": "Action Required",
    "预测器未就绪": "Predictor Not Ready",
    "预测器错误": "Predictor Error",
    "不支持的格式": "Unsupported Format",
    "无提示信息": "No Prompts",
    "精修选项": "Refine Options",
    "模型加载错误": "Model Load Error",
    "无数据": "No Data",
    "未知任务错误": "Unknown Task Error",

    # --- Informational Dialog Paragraphs ---
    "其他 AI 任务正在运行中，请稍后再试。": "Other AI tasks running, please try later.",
    "其他保存操作正在进行中。": "Other save operations in progress.",
    "没有可保存的增强结果。": "No enhanced result to save.",
    "没有有效的图像源可供重新加载。": "No valid image source to reload.",
    "最终蒙版为空。是否保存完全透明的图像？": "Final mask is empty. Save as fully transparent image?",
    "没有有效的图像分割结果可保存。": "No valid segmentation result to save.",
    "请先在画布上选中一个素材。": "Please select an asset on the canvas first.",
    "请点击“添加新对象”图层或选中现有对象以编辑其蒙版。": "Click '+ Add Target' or select existing target to edit mask.",
    "请在右侧点击“添加新对象”，或在画面上点击已有对象的蒙版。": "Click '+ Add Target' on right, or click existing mask on screen.",
    "当前帧与该目标首帧不匹配，是否立刻跳转到第 ": "Current frame mismatches target's first frame, jump to Frame ",
    " 帧？": " ?",
    "是否立刻跳转到首帧？": "Jump to first frame immediately?",
    "您确定要彻底删除【": "Are you sure to delete [",
    "】及其所有视频抠图轨道吗？\n(可通过点击撤回恢复)": "] and all its video matting tracks?\n(Can be restored via Undo)",
    "您确定要清除所有目标对象及其交互点吗？\n这也将清除所有生成的预览和完整抠图结果！": "Clear all targets and interaction points?\nThis clears all previews and matting results!",
    "请在【当前选中的视频片段】内，添加提示点、框或画笔蒙版。": "Add points, boxes or brush strokes within the [currently selected clip].",
    "将对当前选中的片段进行独立隔离处理。\n是否继续？": "Will isolate and process the currently selected clip.\nContinue?",
    "已进入重新编辑模式，请在修改后再次点击开始渲染。": "Re-edit mode entered, click Start Render again after modifying.",
    "抠图完成，但未生成任何有效蒙版。请检查标注点。": "Matting done, but no valid mask generated. Check annotation points.",
    "未加载视频或帧 data 不可用。": "Video not loaded or frame data unavailable.",
    "没有有效的视频抠图结果可保存。请先运行抠图处理。": "No valid video matting result to save. Run matting first.",
    "没有可保存的增强结果。请先应用增强。": "No enhanced result to save. Apply enhancement first.",
    "这个素材已经被高清增强过了。": "This asset has already been enhanced.",
    "请先在画布上单选一个素材。": "Please select a single asset on the canvas first.",
    "您确定要重置当前图像的抠图状态吗？\n\n这将：\n  • 清除所有交互提示 (点/框/绘制)。\n  • 将蒙版恢复到初始图像状态。\n  • 将所有“蒙版优化”设置恢复为默认值。\n  • 清除撤销/重做历史记录。\n\n此操作无法撤销啦！！！": "Are you sure to reset matting state?\n\nThis will:\n  • Clear all prompts (Points/Boxes/Strokes).\n  • Restore mask to initial state.\n  • Reset all 'Mask Adjustments' to default.\n  • Clear Undo/Redo history.\n\nThis cannot be undone!",
    "加载一张新图片将会丢失当前所有的抠图进度。\n您确定要继续吗？": "Loading a new image will lose all current matting progress.\nContinue?",
    "请松开“预览抠图”按钮。": "Please release 'Preview Mask' button.",
    "当前图像分割有未保存的更改。加载新图像将丢弃 these 更改。\n您确定要继续吗？": "Current segmentation has unsaved changes. Loading new image will discard them.\nContinue?",
    "独立加载新视频将清空当前故事板所有片段和抠图结果。\n您确定要继续吗？\n(如需拼接，请点击左侧\"项目库\"的\"添加\"按钮)": "Loading new video independently clears storyboard and matting results.\nContinue?\n(To merge, click '+ Add' in Project Library)",
    "您希望导出的GIF背景是透明的吗？\n\n• 选择“Yes”将生成透明背景的GIF。\n• 选择“No”将使用当前选择的纯色/图片背景。": "Do you want exported GIF to be transparent?\n\n• 'Yes' for transparent background.\n• 'No' to use currently selected solid/image background.",
    "退出前请注意:\n\n": "Before exiting, please note:\n\n",
    "  • 视频抠图有未保存的结果。\n": "  • Unsaved video matting results.\n",
    "  • 创意工坊有未保存的创作。\n": "  • Unsaved creation in Workshop.\n",
    "\n您确定要强制退出吗？ (正在运行的任务将尝试取消)": "\nForce exit? (Running tasks will be cancelled)",
    "所有素材内容都在画布内。请选择保存尺寸：": "All assets within canvas. Choose save size:",
    "按画布尺寸": "Canvas Size",
    "按拼图内容边界": "Content Bounds",
    "检测到有素材超出了画布边界。\n\n为确保内容完整，将自动采用“按内容边界”保存。": "Assets detected outside canvas bounds.\n\nWill auto-save using 'Content Bounds' to ensure completeness.",
    "已达到最大目标对象数量 (": "Max targets reached (",
    "该文件夹内没有找到支持的图像文件。": "No supported images found in this folder.",
    "请选择要增强的图像。": "Please select an image to enhance.",
}


def set_app_lang(lang_code: str):
    """
    Sets the active application language code.
    """
    APP_STATE["LANG"] = lang_code


def get_app_lang() -> str:
    """
    Retrieves the currently active application language code.
    """
    return APP_STATE["LANG"]


def _TR(text: str) -> str:
    """
    Translates the given input string from Chinese into the configured target language.
    If a direct key match is not found in I18N_DICT, it triggers a parsing mechanism
    to scan and substitute identified sub-phrases within complex strings.
    """
    if APP_STATE["LANG"] == "zh" or not text:
        return text
    if text in I18N_DICT:
        return I18N_DICT[text]

    # Process fragmented translations for dynamic strings containing variables and numbers
    en_text = text
    # Sort translation keys descending by length to resolve larger phrases first
    sorted_keys = sorted(I18N_DICT.keys(), key=len, reverse=True)
    for zh_phrase in sorted_keys:
        if zh_phrase in en_text:
            en_text = en_text.replace(zh_phrase, I18N_DICT[zh_phrase])

    return en_text