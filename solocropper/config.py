# SoloCropper
# Copyright (c) 2026 Solo
# Original work by Solo | https://sololo.xyz

import tomllib
from pathlib import Path

from .console import print_error


CONFIG_PATH = Path("config.toml")
DEFAULT_SEG_MODEL_PATH = "models/yolo/yolo26x-seg.pt"
DEFAULT_POSE_MODEL_PATH = "models/yolo/yolo26x-pose.pt"
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".png",
    ".apng",
    ".bmp",
    ".dib",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
    ".avif",
    ".avifs",
    ".ico",
}
SUPPORTED_OUTPUT_FORMATS = {"jpg", "jpeg", "png", "bmp", "webp"}
DEFAULT_SEG_CONF_THRES = 0.45
DEFAULT_POSE_CONF_THRES = 0.45
DEFAULT_SEG_FALLBACK_CONF_THRES = 0.25
DEFAULT_KEYPOINT_CONF_THRES = 0.2
DEFAULT_MAIN_BOX_COLOR = (255, 255, 0, 255)
DEFAULT_ANNOTATION_COLORS = {
    "shoulder": (255, 99, 71, 255),
    "shoulder_only": (255, 20, 147, 255),
    "waist": (50, 205, 50, 255),
    "hip": (30, 144, 255, 255),
    "glute": (255, 140, 0, 255),
    "knee": (186, 85, 211, 255),
    "ankle": (0, 206, 209, 255),
}
DEFAULT_LINE_WIDTH = 5
DEFAULT_LINE_BOX_WIDTH = 4
DEFAULT_LABEL_BG_COLOR = (0, 0, 0, 180)
DEFAULT_LABEL_FONT_SIZE = 28
DEFAULT_LABEL_PADDING_X = 8
DEFAULT_LABEL_PADDING_Y = 4
DEFAULT_LABEL_GAP = 10
DEFAULT_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
]
BOX_SETTING_KEYS = ("full", "shoulder", "shoulder_only", "waist", "hip", "glute", "knee", "ankle")
DEFAULT_SIZE_FIX_LOG_FILENAME = "size_fix_log.txt"
DEFAULT_FLAGGED_CROP_SUBDIR = "_flagged"
DEFAULT_PNG_COMPRESS_LEVEL = 1
DEFAULT_JPEG_QUALITY = 90
DEFAULT_WEBP_QUALITY = 90
DEFAULT_WEBP_METHOD = 0
OVERFLOW_MODE_AUTO = "auto"
OVERFLOW_MODE_CROP = "crop"
OVERFLOW_MODE_PAD = "pad"
VALID_OVERFLOW_MODES = {
    OVERFLOW_MODE_AUTO,
    OVERFLOW_MODE_CROP,
    OVERFLOW_MODE_PAD,
}
DEFAULT_ASPECT_RATIO_OVERFLOW_AUTO_CROP_MAX_LOSS_PERCENT = 20.0
DEFAULT_BOX_OVERFLOW_MODES = {
    "full": OVERFLOW_MODE_AUTO,
    "shoulder": OVERFLOW_MODE_AUTO,
    "shoulder_only": OVERFLOW_MODE_AUTO,
    "waist": OVERFLOW_MODE_AUTO,
    "hip": OVERFLOW_MODE_AUTO,
    "glute": OVERFLOW_MODE_AUTO,
    "knee": OVERFLOW_MODE_AUTO,
    "ankle": OVERFLOW_MODE_AUTO,
}

KEYPOINT_CONF_THRES = DEFAULT_KEYPOINT_CONF_THRES
MAIN_BOX_COLOR = DEFAULT_MAIN_BOX_COLOR
ANNOTATION_COLORS = DEFAULT_ANNOTATION_COLORS.copy()
LINE_WIDTH = DEFAULT_LINE_WIDTH
LINE_BOX_WIDTH = DEFAULT_LINE_BOX_WIDTH
LABEL_BG_COLOR = DEFAULT_LABEL_BG_COLOR
LABEL_FONT_SIZE = DEFAULT_LABEL_FONT_SIZE
LABEL_PADDING_X = DEFAULT_LABEL_PADDING_X
LABEL_PADDING_Y = DEFAULT_LABEL_PADDING_Y
LABEL_GAP = DEFAULT_LABEL_GAP
FONT_CANDIDATES = list(DEFAULT_FONT_CANDIDATES)
SIZE_FIX_LOG_FILENAME = DEFAULT_SIZE_FIX_LOG_FILENAME
FLAGGED_CROP_SUBDIR = DEFAULT_FLAGGED_CROP_SUBDIR
PNG_COMPRESS_LEVEL = DEFAULT_PNG_COMPRESS_LEVEL
JPEG_QUALITY = DEFAULT_JPEG_QUALITY
WEBP_QUALITY = DEFAULT_WEBP_QUALITY
WEBP_METHOD = DEFAULT_WEBP_METHOD


def make_default_box_settings():
    return {
        box_name: {
            "enabled": True,
            "expand": 0,
            "aspect_ratio_fix": [],
            "overflow_mode": DEFAULT_BOX_OVERFLOW_MODES.get(box_name, OVERFLOW_MODE_AUTO),
        }
        for box_name in BOX_SETTING_KEYS
    }


def make_default_models_config():
    return {
        "segmentation": DEFAULT_SEG_MODEL_PATH,
        "pose": DEFAULT_POSE_MODEL_PATH,
    }


def make_default_paths_config():
    return {
        "input_dir": DEFAULT_INPUT_DIR,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "clear_output_dir_on_start": False,
        "size_fix_log_filename": DEFAULT_SIZE_FIX_LOG_FILENAME,
        "flagged_crop_subdir": DEFAULT_FLAGGED_CROP_SUBDIR,
    }


def make_default_thresholds_config():
    return {
        "segmentation": DEFAULT_SEG_CONF_THRES,
        "pose": DEFAULT_POSE_CONF_THRES,
        "segmentation_fallback": DEFAULT_SEG_FALLBACK_CONF_THRES,
        "keypoint": DEFAULT_KEYPOINT_CONF_THRES,
    }


def make_default_annotation_config():
    return {
        "main_box_color": DEFAULT_MAIN_BOX_COLOR,
        "region_colors": DEFAULT_ANNOTATION_COLORS.copy(),
        "line_width": DEFAULT_LINE_WIDTH,
        "line_box_width": DEFAULT_LINE_BOX_WIDTH,
        "label_bg_color": DEFAULT_LABEL_BG_COLOR,
        "label_font_size": DEFAULT_LABEL_FONT_SIZE,
        "label_padding_x": DEFAULT_LABEL_PADDING_X,
        "label_padding_y": DEFAULT_LABEL_PADDING_Y,
        "label_gap": DEFAULT_LABEL_GAP,
        "font_candidates": list(DEFAULT_FONT_CANDIDATES),
    }


def make_default_save_config():
    return {
        "png_compress_level": DEFAULT_PNG_COMPRESS_LEVEL,
        "jpeg_quality": DEFAULT_JPEG_QUALITY,
        "webp_quality": DEFAULT_WEBP_QUALITY,
        "webp_method": DEFAULT_WEBP_METHOD,
    }


def make_default_aspect_ratio_overflow_config():
    return {
        "auto_crop_max_loss_percent": DEFAULT_ASPECT_RATIO_OVERFLOW_AUTO_CROP_MAX_LOSS_PERCENT,
    }


def make_default_crop_logic_config():
    return {
        "size_expand_threshold_percent": 20,
        "upscale_small_outputs": False,
    }


def make_default_config():
    crop_logic = make_default_crop_logic_config()
    return {
        "output_format": "png",
        "box_mode": True,
        "crop_mode": False,
        "device": "cpu",
        "models": make_default_models_config(),
        "paths": make_default_paths_config(),
        "thresholds": make_default_thresholds_config(),
        "annotation": make_default_annotation_config(),
        "save": make_default_save_config(),
        "aspect_ratio_overflow": make_default_aspect_ratio_overflow_config(),
        "crop_logic": crop_logic,
        "box_rules": dict(crop_logic),
        "size_expand_threshold_percent": crop_logic["size_expand_threshold_percent"],
        "upscale_small_outputs": crop_logic["upscale_small_outputs"],
        "box_settings": make_default_box_settings(),
    }


DEFAULT_CONFIG = make_default_config()


def parse_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def normalize_output_format(value):
    normalized = str(value).strip().lower().lstrip(".")
    if normalized in SUPPORTED_OUTPUT_FORMATS:
        return normalized
    return DEFAULT_CONFIG["output_format"]


def normalize_percent(value, default):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return float(default)


def normalize_conf_threshold(value, default):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return float(default)


def normalize_non_negative_int(value, default):
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return int(default)


def normalize_bounded_int(value, default, minimum, maximum):
    try:
        normalized = int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)
    return max(minimum, min(maximum, normalized))


def normalize_path_text(value, default):
    if value is None:
        return str(default)

    text = str(value).strip()
    return text if text else str(default)


def normalize_device(value, default):
    if value is None:
        return str(default)

    text = str(value).strip().lower()
    return text if text else str(default)


def normalize_color(value, default):
    default_color = tuple(default)
    if not isinstance(value, (list, tuple)) or len(value) not in {3, 4}:
        return default_color

    normalized = []
    for channel in value:
        try:
            normalized.append(max(0, min(255, int(round(float(channel))))))
        except (TypeError, ValueError):
            return default_color

    if len(normalized) == 3:
        normalized.append(default_color[3] if len(default_color) >= 4 else 255)

    return tuple(normalized)


def normalize_font_candidates(value, default):
    if not isinstance(value, list):
        return list(default)

    normalized = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)

    return normalized if normalized else list(default)


def normalize_expand_pixels(value):
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return 0


def normalize_overflow_mode(value, default):
    if value is None:
        return str(default)

    normalized = str(value).strip().lower()
    if normalized in VALID_OVERFLOW_MODES:
        return normalized
    return str(default)


def normalize_aspect_ratio_fix_list(value):
    if not isinstance(value, list):
        return []

    normalized = []
    seen = set()
    for item in value:
        if isinstance(item, bool):
            continue
        if not isinstance(item, (str, int, float)):
            continue

        text = str(item).strip().replace(" ", "")
        if not text or text in seen:
            continue

        seen.add(text)
        normalized.append(text)

    return normalized


def normalize_box_settings(value):
    normalized = make_default_box_settings()
    if not isinstance(value, dict):
        return normalized

    for box_name in BOX_SETTING_KEYS:
        box_value = value.get(box_name)
        if not isinstance(box_value, dict):
            continue

        normalized[box_name]["enabled"] = parse_bool(
            box_value.get("enabled", normalized[box_name]["enabled"]),
            normalized[box_name]["enabled"],
        )
        normalized[box_name]["expand"] = normalize_expand_pixels(
            box_value.get("expand", normalized[box_name]["expand"])
        )
        normalized[box_name]["overflow_mode"] = normalize_overflow_mode(
            box_value.get("overflow_mode", normalized[box_name]["overflow_mode"]),
            normalized[box_name]["overflow_mode"],
        )

        aspect_ratio_fix = box_value.get("aspect_ratio_fix", normalized[box_name]["aspect_ratio_fix"])
        normalized[box_name]["aspect_ratio_fix"] = normalize_aspect_ratio_fix_list(aspect_ratio_fix)

    return normalized


def normalize_annotation_region_colors(value):
    normalized = DEFAULT_ANNOTATION_COLORS.copy()
    if not isinstance(value, dict):
        return normalized

    for label, default_color in DEFAULT_ANNOTATION_COLORS.items():
        normalized[label] = normalize_color(value.get(label, default_color), default_color)

    return normalized


def is_box_enabled(box_settings, box_name):
    if not isinstance(box_settings, dict):
        return True
    box_value = box_settings.get(box_name)
    if not isinstance(box_value, dict):
        return True
    return parse_bool(box_value.get("enabled", True), True)


def get_box_expand(box_settings, box_name):
    if not isinstance(box_settings, dict):
        return 0
    box_value = box_settings.get(box_name)
    if not isinstance(box_value, dict):
        return 0
    return normalize_expand_pixels(box_value.get("expand", 0))


def get_box_overflow_mode(box_settings, box_name):
    default_mode = DEFAULT_BOX_OVERFLOW_MODES.get(box_name, OVERFLOW_MODE_AUTO)
    if not isinstance(box_settings, dict):
        return default_mode
    box_value = box_settings.get(box_name)
    if not isinstance(box_value, dict):
        return default_mode
    return normalize_overflow_mode(box_value.get("overflow_mode", default_mode), default_mode)


def parse_positive_int(value):
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def parse_aspect_ratio_spec(value):
    text = str(value).strip().replace(" ", "")
    if not text:
        return None

    if text == "0":
        return {
            "text": "0",
            "suffix": "0",
            "dedupe_key": "none",
            "kind": "none",
            "ratio_width": None,
            "ratio_height": None,
            "resize_width": None,
            "resize_height": None,
        }

    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            return None
        ratio_width = parse_positive_int(parts[0])
        ratio_height = parse_positive_int(parts[1])
        if ratio_width is None or ratio_height is None:
            return None
        return {
            "text": f"{ratio_width}:{ratio_height}",
            "suffix": f"{ratio_width}x{ratio_height}",
            "dedupe_key": f"ratio:{ratio_width}:{ratio_height}",
            "kind": "ratio",
            "ratio_width": ratio_width,
            "ratio_height": ratio_height,
            "resize_width": None,
            "resize_height": None,
        }

    if "," in text:
        parts = text.split(",")
        if len(parts) != 2:
            return None
        resize_width = parse_positive_int(parts[0])
        resize_height = parse_positive_int(parts[1])
        if resize_width is None or resize_height is None:
            return None
        return {
            "text": f"{resize_width},{resize_height}",
            "suffix": f"{resize_width}x{resize_height}",
            "dedupe_key": f"size:{resize_width},{resize_height}",
            "kind": "size",
            "ratio_width": resize_width,
            "ratio_height": resize_height,
            "resize_width": resize_width,
            "resize_height": resize_height,
        }

    return None


def get_box_aspect_ratio_specs(box_settings, box_name):
    default_spec = get_default_aspect_ratio_spec()
    if not isinstance(box_settings, dict):
        return [default_spec]

    box_value = box_settings.get(box_name)
    if not isinstance(box_value, dict):
        return [default_spec]

    raw_specs = normalize_aspect_ratio_fix_list(box_value.get("aspect_ratio_fix", []))
    if not raw_specs:
        raw_specs = ["0"]

    specs = []
    seen = set()
    for raw_spec in raw_specs:
        parsed_spec = parse_aspect_ratio_spec(raw_spec)
        if parsed_spec is None or parsed_spec["dedupe_key"] in seen:
            continue
        seen.add(parsed_spec["dedupe_key"])
        specs.append(parsed_spec)

    return specs if specs else [default_spec]


def get_default_aspect_ratio_spec():
    return parse_aspect_ratio_spec("0")


def get_preview_aspect_ratio_spec(box_spec_map, box_name):
    for aspect_spec in box_spec_map.get(box_name, []):
        if aspect_spec["kind"] != "none":
            return aspect_spec
    return get_default_aspect_ratio_spec()


def load_config(config_path):
    config = make_default_config()

    if not config_path.exists():
        return config

    try:
        with config_path.open("rb") as config_file:
            loaded = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as exc:
        print_error(f"Error: invalid config file format {config_path}: {exc}")
        return None

    if not isinstance(loaded, dict):
        print_error(f"Error: config file root must be a table/object {config_path}")
        return None

    models_config = loaded.get("models", {})
    if not isinstance(models_config, dict):
        models_config = {}

    paths_config = loaded.get("paths", {})
    if not isinstance(paths_config, dict):
        paths_config = {}

    thresholds_config = loaded.get("thresholds", {})
    if not isinstance(thresholds_config, dict):
        thresholds_config = {}

    annotation_config = loaded.get("annotation", {})
    if not isinstance(annotation_config, dict):
        annotation_config = {}

    save_config = loaded.get("save", {})
    if not isinstance(save_config, dict):
        save_config = {}

    aspect_ratio_overflow_config = loaded.get("aspect_ratio_overflow", {})
    if not isinstance(aspect_ratio_overflow_config, dict):
        aspect_ratio_overflow_config = {}

    crop_logic_config = loaded.get("crop_logic", {})
    if not isinstance(crop_logic_config, dict):
        crop_logic_config = {}

    legacy_box_rules_config = loaded.get("box_rules", {})
    if not isinstance(legacy_box_rules_config, dict):
        legacy_box_rules_config = {}

    config["output_format"] = normalize_output_format(
        save_config.get("output_format", loaded.get("output_format", config["output_format"]))
    )
    config["box_mode"] = parse_bool(loaded.get("box_mode", config["box_mode"]), config["box_mode"])
    config["crop_mode"] = parse_bool(loaded.get("crop_mode", config["crop_mode"]), config["crop_mode"])
    config["device"] = normalize_device(loaded.get("device", config["device"]), config["device"])
    config["models"]["segmentation"] = normalize_path_text(
        models_config.get("segmentation", config["models"]["segmentation"]),
        config["models"]["segmentation"],
    )
    config["models"]["pose"] = normalize_path_text(
        models_config.get("pose", config["models"]["pose"]),
        config["models"]["pose"],
    )
    config["paths"]["input_dir"] = normalize_path_text(
        paths_config.get("input_dir", config["paths"]["input_dir"]),
        config["paths"]["input_dir"],
    )
    config["paths"]["output_dir"] = normalize_path_text(
        paths_config.get("output_dir", config["paths"]["output_dir"]),
        config["paths"]["output_dir"],
    )
    config["paths"]["clear_output_dir_on_start"] = parse_bool(
        paths_config.get("clear_output_dir_on_start", config["paths"]["clear_output_dir_on_start"]),
        config["paths"]["clear_output_dir_on_start"],
    )
    config["paths"]["size_fix_log_filename"] = normalize_path_text(
        paths_config.get("size_fix_log_filename", config["paths"]["size_fix_log_filename"]),
        config["paths"]["size_fix_log_filename"],
    )
    config["paths"]["flagged_crop_subdir"] = normalize_path_text(
        paths_config.get("flagged_crop_subdir", config["paths"]["flagged_crop_subdir"]),
        config["paths"]["flagged_crop_subdir"],
    )
    config["thresholds"]["segmentation"] = normalize_conf_threshold(
        thresholds_config.get(
            "segmentation",
            loaded.get("seg_conf_threshold", config["thresholds"]["segmentation"]),
        ),
        config["thresholds"]["segmentation"],
    )
    config["thresholds"]["pose"] = normalize_conf_threshold(
        thresholds_config.get(
            "pose",
            loaded.get("pose_conf_threshold", config["thresholds"]["pose"]),
        ),
        config["thresholds"]["pose"],
    )
    config["thresholds"]["segmentation_fallback"] = normalize_conf_threshold(
        thresholds_config.get(
            "segmentation_fallback",
            loaded.get(
                "seg_fallback_conf_threshold",
                config["thresholds"]["segmentation_fallback"],
            ),
        ),
        config["thresholds"]["segmentation_fallback"],
    )
    config["thresholds"]["keypoint"] = normalize_conf_threshold(
        thresholds_config.get(
            "keypoint",
            loaded.get("keypoint_conf_threshold", config["thresholds"]["keypoint"]),
        ),
        config["thresholds"]["keypoint"],
    )
    config["annotation"]["main_box_color"] = normalize_color(
        annotation_config.get("main_box_color", config["annotation"]["main_box_color"]),
        config["annotation"]["main_box_color"],
    )
    config["annotation"]["region_colors"] = normalize_annotation_region_colors(
        annotation_config.get("region_colors", config["annotation"]["region_colors"])
    )
    config["annotation"]["line_width"] = normalize_non_negative_int(
        annotation_config.get("line_width", config["annotation"]["line_width"]),
        config["annotation"]["line_width"],
    )
    config["annotation"]["line_box_width"] = normalize_non_negative_int(
        annotation_config.get("line_box_width", config["annotation"]["line_box_width"]),
        config["annotation"]["line_box_width"],
    )
    config["annotation"]["label_bg_color"] = normalize_color(
        annotation_config.get("label_bg_color", config["annotation"]["label_bg_color"]),
        config["annotation"]["label_bg_color"],
    )
    config["annotation"]["label_font_size"] = normalize_non_negative_int(
        annotation_config.get("label_font_size", config["annotation"]["label_font_size"]),
        config["annotation"]["label_font_size"],
    )
    config["annotation"]["label_padding_x"] = normalize_non_negative_int(
        annotation_config.get("label_padding_x", config["annotation"]["label_padding_x"]),
        config["annotation"]["label_padding_x"],
    )
    config["annotation"]["label_padding_y"] = normalize_non_negative_int(
        annotation_config.get("label_padding_y", config["annotation"]["label_padding_y"]),
        config["annotation"]["label_padding_y"],
    )
    config["annotation"]["label_gap"] = normalize_non_negative_int(
        annotation_config.get("label_gap", config["annotation"]["label_gap"]),
        config["annotation"]["label_gap"],
    )
    config["annotation"]["font_candidates"] = normalize_font_candidates(
        annotation_config.get("font_candidates", config["annotation"]["font_candidates"]),
        config["annotation"]["font_candidates"],
    )
    config["save"]["png_compress_level"] = normalize_bounded_int(
        save_config.get("png_compress_level", config["save"]["png_compress_level"]),
        config["save"]["png_compress_level"],
        0,
        9,
    )
    config["save"]["jpeg_quality"] = normalize_bounded_int(
        save_config.get("jpeg_quality", config["save"]["jpeg_quality"]),
        config["save"]["jpeg_quality"],
        1,
        100,
    )
    config["save"]["webp_quality"] = normalize_bounded_int(
        save_config.get("webp_quality", config["save"]["webp_quality"]),
        config["save"]["webp_quality"],
        1,
        100,
    )
    config["save"]["webp_method"] = normalize_bounded_int(
        save_config.get("webp_method", config["save"]["webp_method"]),
        config["save"]["webp_method"],
        0,
        6,
    )
    config["aspect_ratio_overflow"]["auto_crop_max_loss_percent"] = normalize_percent(
        aspect_ratio_overflow_config.get(
            "auto_crop_max_loss_percent",
            config["aspect_ratio_overflow"]["auto_crop_max_loss_percent"],
        ),
        config["aspect_ratio_overflow"]["auto_crop_max_loss_percent"],
    )
    config["crop_logic"]["size_expand_threshold_percent"] = normalize_percent(
        crop_logic_config.get(
            "size_expand_threshold_percent",
            legacy_box_rules_config.get(
                "size_expand_threshold_percent",
                loaded.get(
                    "size_expand_threshold_percent",
                    config["crop_logic"]["size_expand_threshold_percent"],
                ),
            ),
        ),
        config["crop_logic"]["size_expand_threshold_percent"],
    )
    config["crop_logic"]["upscale_small_outputs"] = parse_bool(
        crop_logic_config.get(
            "upscale_small_outputs",
            legacy_box_rules_config.get(
                "upscale_small_outputs",
                loaded.get("upscale_small_outputs", config["crop_logic"]["upscale_small_outputs"]),
            ),
        ),
        config["crop_logic"]["upscale_small_outputs"],
    )
    config["box_rules"] = dict(config["crop_logic"])
    config["size_expand_threshold_percent"] = config["crop_logic"]["size_expand_threshold_percent"]
    config["upscale_small_outputs"] = config["crop_logic"]["upscale_small_outputs"]
    config["box_settings"] = normalize_box_settings(loaded.get("box_settings", config["box_settings"]))
    return config


def resolve_config_path(base_dir, configured_path):
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return base_dir / path


def apply_runtime_config(config):
    global KEYPOINT_CONF_THRES
    global MAIN_BOX_COLOR
    global ANNOTATION_COLORS
    global LINE_WIDTH
    global LINE_BOX_WIDTH
    global LABEL_BG_COLOR
    global LABEL_FONT_SIZE
    global LABEL_PADDING_X
    global LABEL_PADDING_Y
    global LABEL_GAP
    global FONT_CANDIDATES
    global SIZE_FIX_LOG_FILENAME
    global FLAGGED_CROP_SUBDIR
    global PNG_COMPRESS_LEVEL
    global JPEG_QUALITY
    global WEBP_QUALITY
    global WEBP_METHOD

    thresholds = config["thresholds"]
    annotation = config["annotation"]
    paths = config["paths"]
    save = config["save"]

    KEYPOINT_CONF_THRES = thresholds["keypoint"]
    MAIN_BOX_COLOR = tuple(annotation["main_box_color"])
    ANNOTATION_COLORS = annotation["region_colors"].copy()
    LINE_WIDTH = annotation["line_width"]
    LINE_BOX_WIDTH = annotation["line_box_width"]
    LABEL_BG_COLOR = tuple(annotation["label_bg_color"])
    LABEL_FONT_SIZE = annotation["label_font_size"]
    LABEL_PADDING_X = annotation["label_padding_x"]
    LABEL_PADDING_Y = annotation["label_padding_y"]
    LABEL_GAP = annotation["label_gap"]
    FONT_CANDIDATES = list(annotation["font_candidates"])
    SIZE_FIX_LOG_FILENAME = paths["size_fix_log_filename"]
    FLAGGED_CROP_SUBDIR = paths["flagged_crop_subdir"]
    PNG_COMPRESS_LEVEL = save["png_compress_level"]
    JPEG_QUALITY = save["jpeg_quality"]
    WEBP_QUALITY = save["webp_quality"]
    WEBP_METHOD = save["webp_method"]


def get_annotation_color(label):
    return ANNOTATION_COLORS.get(label, MAIN_BOX_COLOR)
