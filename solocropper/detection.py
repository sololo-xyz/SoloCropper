# SoloCropper
# Copyright (c) 2026 Solo
# Original work by Solo | https://sololo.xyz

import sys

import numpy as np

from . import config
from .console import print_notice
from .geometry import box_area, box_iou, clamp_box, clamp_line_y, intersection_area, union_boxes

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    missing_name = getattr(exc, "name", "") or "required package"
    if missing_name == "torch":
        print("Error: PyTorch is not installed in this virtual environment.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Quick fix for CPU:", file=sys.stderr)
        print("  pip install torch torchvision", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        print("", file=sys.stderr)
        print("If you want CUDA, install PyTorch from the official selector first:", file=sys.stderr)
        print("  https://pytorch.org/get-started/locally/", file=sys.stderr)
        sys.exit(1)

    if missing_name == "ultralytics":
        print("Error: ultralytics is not installed in this virtual environment.", file=sys.stderr)
        print("Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    raise


PERSON_CLASS_ID = 0
LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 6
LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12
LEFT_KNEE_INDEX = 13
RIGHT_KNEE_INDEX = 14
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16
NOSE_INDEX = 0
AUTO_DEVICE_CPU_FALLBACK_WARNED = False


class DeviceConfigurationError(RuntimeError):
    pass


def is_device_configuration_error(error_text):
    normalized = error_text.lower()
    return any(
        phrase in normalized
        for phrase in (
            "invalid cuda",
            "use 'device=cpu'",
            'use "device=cpu"',
            "torch.cuda.is_available(): false",
            "no cuda gpus are available",
            "cuda is not available",
            "unknown or unsupported compute capability",
        )
    )


def get_model_input_image(source_image):
    return np.asarray(source_image.convert("RGB"))


def predict_detection_result(
    model,
    source_image,
    conf_threshold,
    device,
    retina_masks=False,
    class_ids=None,
):
    global AUTO_DEVICE_CPU_FALLBACK_WARNED

    predict_kwargs = {
        "source": source_image,
        "conf": conf_threshold,
        "classes": list(class_ids) if class_ids is not None else None,
        "verbose": False,
    }
    if device != "auto":
        predict_kwargs["device"] = device
    if retina_masks:
        predict_kwargs["retina_masks"] = True
    try:
        results = model.predict(**predict_kwargs)
    except Exception as exc:
        error_message = str(exc)
        error_text = error_message.lower()
        is_auto_cuda_failure = (
            device == "auto"
            and (
                "cuda error" in error_text
                or "no kernel image is available" in error_text
                or "not compatible with the current pytorch installation" in error_text
                or "cuda capability" in error_text
            )
        )
        if not is_auto_cuda_failure:
            if is_device_configuration_error(error_message):
                configured_device = device if device != "auto" else "auto"
                raise DeviceConfigurationError(
                    f"device setting '{configured_device}' is not usable in the current environment. "
                    f"Original error: {error_message} "
                    'Set device="cpu" to run on CPU, or install a compatible CUDA-enabled PyTorch build.'
                ) from exc
            raise

        if not AUTO_DEVICE_CPU_FALLBACK_WARNED:
            print_notice(
                "Notice: auto device selection failed on CUDA, retrying on CPU. "
                "Install a newer CUDA-compatible PyTorch build if you want GPU inference."
            )
            AUTO_DEVICE_CPU_FALLBACK_WARNED = True

        predict_kwargs["device"] = "cpu"
        results = model.predict(**predict_kwargs)
    return results[0] if results else None


def predict_person_result(model, source_image, conf_threshold, device, retina_masks=False):
    return predict_detection_result(
        model,
        source_image,
        conf_threshold,
        device,
        retina_masks=retina_masks,
        class_ids=[PERSON_CLASS_ID],
    )


def get_mask_box(result, index, image_width, image_height):
    if result.masks is None or not hasattr(result.masks, "xy") or index >= len(result.masks.xy):
        return None

    segments = result.masks.xy[index]
    if isinstance(segments, np.ndarray):
        segments = [segments]

    min_x = None
    min_y = None
    max_x = None
    max_y = None

    for segment in segments:
        points = np.asarray(segment, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            continue

        segment_min_x = float(points[:, 0].min())
        segment_min_y = float(points[:, 1].min())
        segment_max_x = float(points[:, 0].max())
        segment_max_y = float(points[:, 1].max())

        min_x = segment_min_x if min_x is None else min(min_x, segment_min_x)
        min_y = segment_min_y if min_y is None else min(min_y, segment_min_y)
        max_x = segment_max_x if max_x is None else max(max_x, segment_max_x)
        max_y = segment_max_y if max_y is None else max(max_y, segment_max_y)

    if min_x is None:
        return None

    return clamp_box((min_x, min_y, max_x + 1, max_y + 1), image_width, image_height)


def choose_main_detection(result, class_id=None):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None, 0

    image_height, image_width = result.orig_shape
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    candidates = []
    for index in range(len(boxes_xyxy)):
        if class_id is not None and int(classes[index]) != class_id:
            continue

        detection_box = clamp_box(boxes_xyxy[index], image_width, image_height)
        mask_box = get_mask_box(result, index, image_width, image_height)
        final_box = union_boxes(detection_box, mask_box)
        candidates.append(
            {
                "index": index,
                "box": final_box,
                "area": box_area(final_box),
            }
        )

    if not candidates:
        return None, 0

    main_detection = max(candidates, key=lambda item: item["area"])
    return main_detection, len(candidates)


def choose_main_person(result):
    return choose_main_detection(result, class_id=PERSON_CLASS_ID)


def choose_main_object(result):
    return choose_main_detection(result, class_id=None)


def choose_pose_person(result, target_box):
    if result is None or result.boxes is None or len(result.boxes) == 0 or target_box is None:
        return None

    image_height, image_width = result.orig_shape
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    target_center_x = (target_box[0] + target_box[2]) / 2.0
    target_center_y = (target_box[1] + target_box[3]) / 2.0

    candidates = []
    for index in range(len(boxes_xyxy)):
        if int(classes[index]) != PERSON_CLASS_ID:
            continue

        pose_box = clamp_box(boxes_xyxy[index], image_width, image_height)
        pose_center_x = (pose_box[0] + pose_box[2]) / 2.0
        pose_center_y = (pose_box[1] + pose_box[3]) / 2.0
        distance_sq = (pose_center_x - target_center_x) ** 2 + (pose_center_y - target_center_y) ** 2

        candidates.append(
            {
                "index": index,
                "iou": box_iou(pose_box, target_box),
                "overlap": intersection_area(pose_box, target_box),
                "distance_sq": distance_sq,
            }
        )

    if not candidates:
        return None

    best_match = max(candidates, key=lambda item: (item["iou"], item["overlap"], -item["distance_sq"]))
    return best_match["index"]


def get_pair_line_y(result, person_index, left_keypoint_index, right_keypoint_index, image_height):
    if result is None or result.keypoints is None:
        return None

    keypoints_xy = result.keypoints.xy.cpu().numpy()
    if person_index >= len(keypoints_xy):
        return None

    keypoints_conf = None
    if result.keypoints.conf is not None:
        keypoints_conf = result.keypoints.conf.cpu().numpy()

    candidate_ys = []
    for keypoint_index in (left_keypoint_index, right_keypoint_index):
        point = keypoints_xy[person_index][keypoint_index]
        x = float(point[0])
        y = float(point[1])

        if np.isnan(x) or np.isnan(y):
            continue

        if keypoints_conf is not None:
            confidence = float(keypoints_conf[person_index][keypoint_index])
            if confidence < config.KEYPOINT_CONF_THRES:
                continue

        if x <= 0 and y <= 0:
            continue

        candidate_ys.append(y)

    if not candidate_ys:
        return None

    return max(0, min(int(round(max(candidate_ys))), image_height - 1))


def get_valid_keypoint(result, person_index, keypoint_index):
    if result is None or result.keypoints is None:
        return None

    keypoints_xy = result.keypoints.xy.cpu().numpy()
    if person_index >= len(keypoints_xy):
        return None

    point = keypoints_xy[person_index][keypoint_index]
    x = float(point[0])
    y = float(point[1])
    if np.isnan(x) or np.isnan(y):
        return None

    if result.keypoints.conf is not None:
        keypoints_conf = result.keypoints.conf.cpu().numpy()
        confidence = float(keypoints_conf[person_index][keypoint_index])
        if confidence < config.KEYPOINT_CONF_THRES:
            return None

    if x <= 0 and y <= 0:
        return None

    return (x, y)


def get_mask_array(result, index):
    if result is None or index is None or result.masks is None or not hasattr(result.masks, "data"):
        return None

    if index >= len(result.masks.data):
        return None

    mask = result.masks.data[index].cpu().numpy()
    if mask.ndim != 2:
        return None

    return mask > 0.5


def get_mask_region_box(mask, top_y, bottom_y, image_width, image_height, fallback_box=None):
    top_y = clamp_line_y(top_y, image_height)
    bottom_y = clamp_line_y(bottom_y, image_height)
    if bottom_y < top_y:
        top_y, bottom_y = bottom_y, top_y

    if mask is None or mask.size == 0:
        if fallback_box is None:
            return None
        return clamp_box((fallback_box[0], top_y, fallback_box[2], bottom_y + 1), image_width, image_height)

    mask_height, mask_width = mask.shape
    if image_height <= 1:
        row_start = 0
        row_end = 0
    else:
        row_start = int(round(float(top_y) * (mask_height - 1) / (image_height - 1)))
        row_end = int(round(float(bottom_y) * (mask_height - 1) / (image_height - 1)))

    row_start = max(0, min(row_start, mask_height - 1))
    row_end = max(0, min(row_end, mask_height - 1))
    if row_end < row_start:
        row_start, row_end = row_end, row_start

    region = mask[row_start : row_end + 1]
    if region.size == 0:
        if fallback_box is None:
            return None
        return clamp_box((fallback_box[0], top_y, fallback_box[2], bottom_y + 1), image_width, image_height)

    active_columns = np.flatnonzero(region.any(axis=0))
    if active_columns.size == 0:
        if fallback_box is None:
            return None
        return clamp_box((fallback_box[0], top_y, fallback_box[2], bottom_y + 1), image_width, image_height)

    left_column = int(active_columns[0])
    right_column = int(active_columns[-1])

    if image_width <= 1 or mask_width <= 0:
        left = 0
        right = image_width
    else:
        left = int(np.floor(left_column * image_width / mask_width))
        right = int(np.ceil((right_column + 1) * image_width / mask_width))

    if fallback_box is not None:
        left = max(fallback_box[0], left)
        right = min(fallback_box[2], right)

    return clamp_box((left, top_y, right, bottom_y + 1), image_width, image_height)


def get_shoulder_only_region_box(result, person_index, fallback_box, image_width, image_height):
    if result is None or person_index is None or fallback_box is None:
        return None

    shoulder_line_y = get_pair_line_y(
        result,
        person_index,
        LEFT_SHOULDER_INDEX,
        RIGHT_SHOULDER_INDEX,
        image_height,
    )
    if shoulder_line_y is None:
        return None

    shoulder_points = []
    for keypoint_index in (LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX):
        point = get_valid_keypoint(result, person_index, keypoint_index)
        if point is None:
            return None
        shoulder_points.append(point)

    min_shoulder_x = min(float(point[0]) for point in shoulder_points)
    max_shoulder_x = max(float(point[0]) for point in shoulder_points)

    return clamp_box(
        (
            min_shoulder_x,
            fallback_box[1],
            max_shoulder_x + 1,
            shoulder_line_y + 1,
        ),
        image_width,
        image_height,
    )


def get_mask_row_info(mask, y, image_height):
    if mask is None or mask.size == 0:
        return None

    mask_height = mask.shape[0]
    if image_height <= 1:
        row_index = 0
    else:
        row_index = int(round(float(y) * (mask_height - 1) / (image_height - 1)))
    row_index = max(0, min(row_index, mask_height - 1))

    row = mask[row_index]
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return None

    segments = []
    start_x = int(xs[0])
    prev_x = int(xs[0])
    for x in xs[1:]:
        current_x = int(x)
        if current_x == prev_x + 1:
            prev_x = current_x
            continue
        segments.append((start_x, prev_x))
        start_x = current_x
        prev_x = current_x
    segments.append((start_x, prev_x))

    width = int(xs[-1] - xs[0] + 1)
    min_segment_width = max(2, int(round(width * 0.08)))
    significant_segments = [
        segment for segment in segments if (segment[1] - segment[0] + 1) >= min_segment_width
    ]

    return {
        "width": width,
        "segment_count": len(significant_segments) if significant_segments else len(segments),
    }


def get_smoothed_mask_row_info(mask, y, image_height, radius=2):
    widths = []
    segment_count = 0

    for offset in range(-radius, radius + 1):
        sample_y = clamp_line_y(y + offset, image_height)
        row_info = get_mask_row_info(mask, sample_y, image_height)
        if row_info is None:
            continue
        widths.append(float(row_info["width"]))
        segment_count = max(segment_count, int(row_info["segment_count"]))

    if not widths:
        return None

    return {
        "width": sum(widths) / len(widths),
        "segment_count": segment_count,
    }


def get_waist_line_y(result, person_index, image_height):
    shoulder_line_y = get_pair_line_y(
        result, person_index, LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, image_height
    )
    hip_line_y = get_pair_line_y(
        result, person_index, LEFT_HIP_INDEX, RIGHT_HIP_INDEX, image_height
    )

    if shoulder_line_y is None or hip_line_y is None:
        return None

    return clamp_line_y(
        shoulder_line_y + ((hip_line_y - shoulder_line_y) * 0.6),
        image_height,
    )


def get_vertical_compression_ratio(result, person_index, box, image_height):
    if box is None:
        return None

    hip_line_y = get_pair_line_y(
        result, person_index, LEFT_HIP_INDEX, RIGHT_HIP_INDEX, image_height
    )
    if hip_line_y is None:
        return None

    nose_point = get_valid_keypoint(result, person_index, NOSE_INDEX)
    if nose_point is not None:
        top_anchor_y = nose_point[1]
    else:
        shoulder_line_y = get_pair_line_y(
            result, person_index, LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, image_height
        )
        if shoulder_line_y is not None:
            top_anchor_y = shoulder_line_y
        else:
            top_anchor_y = box[1]

    box_height = max(1, box[3] - box[1])
    return max(0.0, (hip_line_y - top_anchor_y) / box_height)


def get_hip_knee_leveling_ratio(result, person_index, box, image_height):
    if box is None:
        return None

    hip_line_y = get_pair_line_y(
        result, person_index, LEFT_HIP_INDEX, RIGHT_HIP_INDEX, image_height
    )
    knee_line_y = get_pair_line_y(
        result, person_index, LEFT_KNEE_INDEX, RIGHT_KNEE_INDEX, image_height
    )
    if hip_line_y is None or knee_line_y is None:
        return None

    box_height = max(1, box[3] - box[1])
    return abs(knee_line_y - hip_line_y) / box_height


def analyze_mask_support(seg_result, seg_person_index, box, hip_line_y, knee_line_y, image_height):
    mask = get_mask_array(seg_result, seg_person_index)
    if mask is None or box is None or hip_line_y is None:
        return None

    hip_info = get_smoothed_mask_row_info(mask, hip_line_y, image_height)
    if hip_info is None or hip_info["width"] <= 0:
        return None

    box_height = max(1, box[3] - box[1])
    search_top = hip_line_y + max(4, int(round(0.02 * box_height)))
    search_bottom = hip_line_y + max(10, int(round(0.14 * box_height)))
    if knee_line_y is not None and knee_line_y > hip_line_y:
        search_bottom = min(search_bottom, hip_line_y + max(6, int(round(0.55 * (knee_line_y - hip_line_y)))))
    search_bottom = min(search_bottom, box[3] - 1)

    if search_bottom <= search_top:
        return None

    width_ratios = []
    continuous_rows = 0
    split_rows = 0
    valid_rows = 0

    step = max(1, int(round((search_bottom - search_top) / 12)))
    for y in range(search_top, search_bottom + 1, step):
        row_info = get_smoothed_mask_row_info(mask, y, image_height)
        if row_info is None:
            continue
        valid_rows += 1
        width_ratios.append(row_info["width"] / max(1.0, hip_info["width"]))
        if row_info["segment_count"] <= 1:
            continuous_rows += 1
        if row_info["segment_count"] >= 2:
            split_rows += 1

    if valid_rows == 0:
        return None

    return {
        "avg_width_ratio": sum(width_ratios) / len(width_ratios),
        "continuous_ratio": continuous_rows / valid_rows,
        "split_ratio": split_rows / valid_rows,
        "valid_rows": valid_rows,
    }


def get_joint_angle(point_a, joint_point, point_c):
    if point_a is None or joint_point is None or point_c is None:
        return None

    vector_a = np.asarray(
        [float(point_a[0]) - float(joint_point[0]), float(point_a[1]) - float(joint_point[1])],
        dtype=np.float32,
    )
    vector_c = np.asarray(
        [float(point_c[0]) - float(joint_point[0]), float(point_c[1]) - float(joint_point[1])],
        dtype=np.float32,
    )

    norm_a = float(np.linalg.norm(vector_a))
    norm_c = float(np.linalg.norm(vector_c))
    if norm_a <= 1e-6 or norm_c <= 1e-6:
        return None

    cosine = float(np.dot(vector_a, vector_c) / (norm_a * norm_c))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def classify_pose_side(result, person_index, shoulder_index, hip_index, knee_index, ankle_index):
    shoulder = get_valid_keypoint(result, person_index, shoulder_index)
    hip = get_valid_keypoint(result, person_index, hip_index)
    knee = get_valid_keypoint(result, person_index, knee_index)
    ankle = get_valid_keypoint(result, person_index, ankle_index)

    hip_angle = get_joint_angle(shoulder, hip, knee)
    knee_angle = get_joint_angle(hip, knee, ankle)

    standing_score = 0.0
    sitting_score = 0.0
    evidence_count = 0

    if hip_angle is not None:
        evidence_count += 1
        if hip_angle >= 170.0:
            standing_score += 0.8
        elif hip_angle >= 155.0:
            standing_score += 0.4
        elif hip_angle <= 110.0:
            sitting_score += 0.8
        elif hip_angle <= 130.0:
            sitting_score += 0.4

    if knee_angle is not None:
        evidence_count += 1
        if knee_angle >= 170.0:
            standing_score += 0.8
        elif knee_angle >= 155.0:
            standing_score += 0.4
        elif knee_angle <= 100.0:
            sitting_score += 0.8
        elif knee_angle <= 125.0:
            sitting_score += 0.4

    if hip_angle is not None and knee_angle is not None:
        evidence_count += 1
        if hip_angle >= 160.0 and knee_angle >= 165.0:
            standing_score += 0.5
        elif hip_angle <= 125.0 and knee_angle <= 135.0:
            sitting_score += 0.5

    return {
        "standing": standing_score,
        "sitting": sitting_score,
        "evidence": evidence_count,
        "hip_angle": hip_angle,
        "knee_angle": knee_angle,
    }


def classify_pose_type(result, person_index, seg_result=None, seg_person_index=None, box=None):
    if result is None or person_index is None or box is None:
        return "unknown"

    image_height = result.orig_shape[0]
    hip_line_y = get_pair_line_y(
        result, person_index, LEFT_HIP_INDEX, RIGHT_HIP_INDEX, image_height
    )
    knee_line_y = get_pair_line_y(
        result, person_index, LEFT_KNEE_INDEX, RIGHT_KNEE_INDEX, image_height
    )
    ankle_line_y = get_pair_line_y(
        result, person_index, LEFT_ANKLE_INDEX, RIGHT_ANKLE_INDEX, image_height
    )

    left_side = classify_pose_side(
        result,
        person_index,
        LEFT_SHOULDER_INDEX,
        LEFT_HIP_INDEX,
        LEFT_KNEE_INDEX,
        LEFT_ANKLE_INDEX,
    )
    right_side = classify_pose_side(
        result,
        person_index,
        RIGHT_SHOULDER_INDEX,
        RIGHT_HIP_INDEX,
        RIGHT_KNEE_INDEX,
        RIGHT_ANKLE_INDEX,
    )

    standing_score = left_side["standing"] + right_side["standing"]
    sitting_score = left_side["sitting"] + right_side["sitting"]
    evidence_count = left_side["evidence"] + right_side["evidence"]

    vertical_compression_ratio = get_vertical_compression_ratio(result, person_index, box, image_height)
    if vertical_compression_ratio is not None:
        evidence_count += 1
        if vertical_compression_ratio >= 0.52:
            sitting_score += 0.8
        elif vertical_compression_ratio >= 0.47:
            sitting_score += 0.4
        elif vertical_compression_ratio <= 0.34:
            standing_score += 0.6

    hip_knee_ratio = get_hip_knee_leveling_ratio(result, person_index, box, image_height)
    if hip_knee_ratio is not None:
        evidence_count += 1
        if hip_knee_ratio <= 0.12:
            sitting_score += 3.2
        elif hip_knee_ratio <= 0.16:
            sitting_score += 2.6
        elif hip_knee_ratio <= 0.20:
            sitting_score += 1.6
        elif hip_knee_ratio >= 0.30:
            standing_score += 3.0
        elif hip_knee_ratio >= 0.24:
            standing_score += 2.2
        elif hip_knee_ratio >= 0.20:
            standing_score += 1.2

    support_metrics = analyze_mask_support(
        seg_result,
        seg_person_index,
        box,
        hip_line_y,
        knee_line_y,
        image_height,
    )
    if support_metrics is not None:
        evidence_count += 1
        if (
            support_metrics["avg_width_ratio"] >= 1.08
            and support_metrics["continuous_ratio"] >= 0.75
        ):
            sitting_score += 2.2
        elif (
            support_metrics["avg_width_ratio"] >= 0.92
            and support_metrics["continuous_ratio"] >= 0.70
        ):
            sitting_score += 1.0

        if (
            support_metrics["avg_width_ratio"] <= 0.78
            or support_metrics["split_ratio"] >= 0.55
        ):
            standing_score += 1.6
        elif (
            support_metrics["avg_width_ratio"] <= 0.88
            and support_metrics["split_ratio"] >= 0.35
        ):
            standing_score += 0.8

    if (
        hip_knee_ratio is not None
        and support_metrics is not None
        and hip_knee_ratio <= 0.28
        and support_metrics["avg_width_ratio"] >= 1.15
        and support_metrics["continuous_ratio"] >= 0.9
    ):
        sitting_score += 1.0

    if ankle_line_y is None and support_metrics is not None:
        if support_metrics["avg_width_ratio"] >= 1.02 and support_metrics["continuous_ratio"] >= 0.8:
            sitting_score += 1.0

    if evidence_count == 0:
        return "unknown"

    if standing_score >= sitting_score + 1.0:
        return "standing"
    if sitting_score >= standing_score + 1.0:
        return "sitting"

    return "unknown"


def get_glute_line_y(pose_result, pose_person_index, seg_result, seg_person_index, box, pose_type):
    if box is None:
        return None

    image_height = None
    if seg_result is not None:
        image_height, _ = seg_result.orig_shape
    elif pose_result is not None:
        image_height, _ = pose_result.orig_shape
    else:
        return None

    hip_line_y = get_pair_line_y(
        pose_result, pose_person_index, LEFT_HIP_INDEX, RIGHT_HIP_INDEX, image_height
    )
    if hip_line_y is None:
        return None

    shoulder_line_y = get_pair_line_y(
        pose_result, pose_person_index, LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, image_height
    )
    knee_line_y = get_pair_line_y(
        pose_result, pose_person_index, LEFT_KNEE_INDEX, RIGHT_KNEE_INDEX, image_height
    )

    if pose_type == "standing" and knee_line_y is not None and knee_line_y > hip_line_y:
        leg_span = knee_line_y - hip_line_y
        pose_estimate = hip_line_y + (0.18 * leg_span)
        search_top = hip_line_y + max(2, int(round(0.03 * leg_span)))
        search_bottom = hip_line_y + max(6, int(round(0.32 * leg_span)))
        search_bottom = min(search_bottom, knee_line_y - 1)
        distance_weight = 0.45
        split_bonus_weight = 0.10
    elif pose_type == "sitting" and knee_line_y is not None:
        return clamp_line_y((hip_line_y + knee_line_y) / 2.0, image_height)
    elif knee_line_y is not None and knee_line_y > hip_line_y:
        leg_span = knee_line_y - hip_line_y
        pose_estimate = hip_line_y + (0.22 * leg_span)
        search_top = hip_line_y + max(2, int(round(0.05 * leg_span)))
        search_bottom = hip_line_y + max(6, int(round(0.45 * leg_span)))
        search_bottom = min(search_bottom, knee_line_y - 1)
        distance_weight = 0.35
        split_bonus_weight = 0.18
    elif shoulder_line_y is not None and hip_line_y > shoulder_line_y:
        torso_span = hip_line_y - shoulder_line_y
        if pose_type == "sitting":
            pose_estimate = hip_line_y + (0.08 * torso_span)
            search_top = hip_line_y + max(2, int(round(0.02 * torso_span)))
            search_bottom = hip_line_y + max(5, int(round(0.16 * torso_span)))
            distance_weight = 0.20
            split_bonus_weight = 0.30
        elif pose_type == "standing":
            pose_estimate = hip_line_y + (0.10 * torso_span)
            search_top = hip_line_y + max(2, int(round(0.03 * torso_span)))
            search_bottom = hip_line_y + max(6, int(round(0.22 * torso_span)))
            distance_weight = 0.45
            split_bonus_weight = 0.10
        else:
            pose_estimate = hip_line_y + (0.12 * torso_span)
            search_top = hip_line_y + max(2, int(round(0.03 * torso_span)))
            search_bottom = hip_line_y + max(6, int(round(0.25 * torso_span)))
            distance_weight = 0.35
            split_bonus_weight = 0.18
    else:
        box_height = box[3] - box[1]
        if pose_type == "sitting":
            pose_estimate = hip_line_y + (0.05 * box_height)
            search_top = hip_line_y + max(2, int(round(0.01 * box_height)))
            search_bottom = hip_line_y + max(6, int(round(0.09 * box_height)))
            distance_weight = 0.20
            split_bonus_weight = 0.30
        elif pose_type == "standing":
            pose_estimate = hip_line_y + (0.07 * box_height)
            search_top = hip_line_y + max(2, int(round(0.02 * box_height)))
            search_bottom = hip_line_y + max(8, int(round(0.12 * box_height)))
            distance_weight = 0.45
            split_bonus_weight = 0.10
        else:
            pose_estimate = hip_line_y + (0.08 * box_height)
            search_top = hip_line_y + max(2, int(round(0.02 * box_height)))
            search_bottom = hip_line_y + max(8, int(round(0.12 * box_height)))
            distance_weight = 0.35
            split_bonus_weight = 0.18

    pose_estimate = clamp_line_y(pose_estimate, image_height)
    search_top = clamp_line_y(max(search_top, hip_line_y), image_height)
    search_bottom = clamp_line_y(min(search_bottom, box[3]), image_height)
    if search_bottom < search_top:
        return pose_estimate

    mask = get_mask_array(seg_result, seg_person_index)
    if mask is None:
        return pose_estimate

    hip_row_info = get_smoothed_mask_row_info(mask, hip_line_y, image_height)
    hip_width = hip_row_info["width"] if hip_row_info is not None else None
    search_range = max(1, search_bottom - search_top)
    best_y = None
    best_score = None

    for y in range(search_top, search_bottom + 1):
        row_info = get_smoothed_mask_row_info(mask, y, image_height)
        if row_info is None:
            continue

        width_reference = hip_width if hip_width is not None else row_info["width"]
        width_reference = max(width_reference, 1.0)
        width_score = row_info["width"] / width_reference
        distance_score = abs(y - pose_estimate) / search_range
        split_bonus = split_bonus_weight if row_info["segment_count"] >= 2 else 0.0
        score = width_score + (distance_weight * distance_score) - split_bonus

        if best_score is None or score < best_score:
            best_score = score
            best_y = y

    return best_y if best_y is not None else pose_estimate


def collect_horizontal_lines(result, person_index, box, pose_type, seg_result=None, seg_person_index=None):
    if box is None:
        return []

    if result is None or person_index is None:
        return []

    image_height, _ = result.orig_shape
    shoulder_line_y = get_pair_line_y(result, person_index, LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, image_height)
    raw_lines = [
        ("shoulder", shoulder_line_y),
        ("shoulder_only", shoulder_line_y),
        ("waist", get_waist_line_y(result, person_index, image_height)),
        ("hip", get_pair_line_y(result, person_index, LEFT_HIP_INDEX, RIGHT_HIP_INDEX, image_height)),
        ("glute", get_glute_line_y(result, person_index, seg_result, seg_person_index, box, pose_type)),
        ("knee", get_pair_line_y(result, person_index, LEFT_KNEE_INDEX, RIGHT_KNEE_INDEX, image_height)),
        ("ankle", get_pair_line_y(result, person_index, LEFT_ANKLE_INDEX, RIGHT_ANKLE_INDEX, image_height)),
    ]

    visible_lines = []
    for label, line_y in raw_lines:
        if line_y is None:
            continue
        if box[1] <= line_y <= box[3]:
            visible_lines.append((label, line_y))

    return visible_lines


def build_region_items(
    lines,
    box,
    image_width,
    image_height,
    seg_result=None,
    seg_person_index=None,
    pose_result=None,
    pose_person_index=None,
):
    mask = get_mask_array(seg_result, seg_person_index)
    region_items = []

    for label, line_y in lines:
        if label == "shoulder_only":
            line_box = get_shoulder_only_region_box(
                pose_result,
                pose_person_index,
                box,
                image_width,
                image_height,
            )
        else:
            line_box = get_mask_region_box(
                mask,
                box[1],
                line_y,
                image_width,
                image_height,
                fallback_box=box,
            )
        region_items.append(
            {
                "label": label,
                "line_y": line_y,
                "box": line_box,
                "color": config.get_annotation_color(label),
            }
        )

    return region_items


def filter_region_items(region_items, box_settings):
    return [item for item in region_items if config.is_box_enabled(box_settings, item["label"])]
