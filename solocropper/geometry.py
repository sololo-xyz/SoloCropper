# SoloCropper
# Copyright (c) 2026 Solo
# Original work by Solo | https://sololo.xyz

from PIL import Image

from .config import normalize_expand_pixels


def clamp_box(box, image_width, image_height):
    x1, y1, x2, y2 = box
    left = max(0, min(int(round(float(x1))), max(image_width - 1, 0)))
    top = max(0, min(int(round(float(y1))), max(image_height - 1, 0)))
    right = max(left + 1, min(int(round(float(x2))), image_width))
    bottom = max(top + 1, min(int(round(float(y2))), image_height))
    return (left, top, right, bottom)


def union_boxes(box_a, box_b):
    if box_a is None:
        return box_b
    if box_b is None:
        return box_a
    return (
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    )


def box_area(box):
    return max(1, box[2] - box[0]) * max(1, box[3] - box[1])


def intersection_area(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def box_iou(box_a, box_b):
    overlap = intersection_area(box_a, box_b)
    if overlap <= 0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def clamp_line_y(line_y, image_height):
    return max(0, min(int(round(float(line_y))), image_height - 1))


def expand_axis_by_pixels(start, end, expand_pixels, max_length):
    start = int(round(float(start)))
    end = int(round(float(end)))
    expand_pixels = normalize_expand_pixels(expand_pixels)
    if expand_pixels <= 0:
        return (start, end)

    requested_start_expand = expand_pixels // 2
    requested_end_expand = expand_pixels - requested_start_expand
    available_start_expand = max(0, start)
    available_end_expand = max(0, max_length - end)

    if (
        available_start_expand >= requested_start_expand
        and available_end_expand >= requested_end_expand
    ):
        return (start - requested_start_expand, end + requested_end_expand)

    symmetric_expand = min(
        available_start_expand,
        available_end_expand,
        expand_pixels // 2,
    )
    return (start - symmetric_expand, end + symmetric_expand)


def expand_box_by_pixels(box, expand_pixels, image_width, image_height):
    if box is None:
        return None

    left, top, right, bottom = box
    expanded_left, expanded_right = expand_axis_by_pixels(left, right, expand_pixels, image_width)
    expanded_top, expanded_bottom = expand_axis_by_pixels(top, bottom, expand_pixels, image_height)
    return clamp_box((expanded_left, expanded_top, expanded_right, expanded_bottom), image_width, image_height)


def expand_axis_to_length(start, end, target_length, max_length):
    current_length = max(1, end - start)
    target_length = max(current_length, int(target_length))
    if target_length <= current_length:
        return (int(start), int(end))

    delta = target_length - current_length
    expanded_start = int(start) - (delta // 2)
    expanded_end = int(end) + (delta - (delta // 2))

    if expanded_start < 0 and expanded_end <= max_length:
        shift = -expanded_start
        expanded_start += shift
        expanded_end += shift
    elif expanded_end > max_length and expanded_start >= 0:
        shift = expanded_end - max_length
        expanded_start -= shift
        expanded_end -= shift

    return (int(expanded_start), int(expanded_end))


def expand_axis_to_length_unbounded(start, end, target_length):
    current_length = max(1, int(end) - int(start))
    target_length = max(current_length, int(target_length))
    if target_length <= current_length:
        return (int(start), int(end))

    delta = target_length - current_length
    expanded_start = int(start) - (delta // 2)
    expanded_end = int(end) + (delta - (delta // 2))
    return (int(expanded_start), int(expanded_end))


def shift_axis_into_image(start, end, max_length):
    shifted_start = int(round(start))
    shifted_end = int(round(end))

    if shifted_start < 0:
        shift = -shifted_start
        shifted_start += shift
        shifted_end += shift

    if shifted_end > max_length:
        shift = shifted_end - max_length
        shifted_start -= shift
        shifted_end -= shift

    return (shifted_start, shifted_end)


def shift_box_into_image(box, image_width, image_height):
    if box is None:
        return None

    left, top, right, bottom = box
    shifted_left, shifted_right = shift_axis_into_image(left, right, image_width)
    shifted_top, shifted_bottom = shift_axis_into_image(top, bottom, image_height)
    return (shifted_left, shifted_top, shifted_right, shifted_bottom)


def get_box_dimensions(box):
    if box is None:
        return (1, 1)

    left, top, right, bottom = [int(round(float(value))) for value in box]
    return (max(1, right - left), max(1, bottom - top))


def would_ratio_correction_overflow_image(box, ratio_width, ratio_height, image_width, image_height):
    if box is None or ratio_width is None or ratio_height is None:
        return False

    box_width, box_height = get_box_dimensions(box)
    current_ratio_scaled = box_width * ratio_height
    target_ratio_scaled = box_height * ratio_width

    if current_ratio_scaled == target_ratio_scaled:
        return False

    if current_ratio_scaled < target_ratio_scaled:
        target_width = (box_height * ratio_width + ratio_height - 1) // ratio_height
        return target_width > image_width

    target_height = (box_width * ratio_height + ratio_width - 1) // ratio_width
    return target_height > image_height


def shrink_axis_to_length(start, end, target_length):
    current_length = max(1, int(end) - int(start))
    target_length = max(1, min(current_length, int(target_length)))
    if target_length >= current_length:
        return (int(start), int(end))

    delta = current_length - target_length
    shrink_start = delta // 2
    shrink_end = delta - shrink_start
    return (int(start) + shrink_start, int(end) - shrink_end)


def shrink_axis_to_length_with_min(start, end, target_length, min_length):
    current_length = max(1, int(end) - int(start))
    min_length = max(1, min(current_length, int(min_length)))
    target_length = max(min_length, min(current_length, int(target_length)))
    if target_length >= current_length:
        return (int(start), int(end))

    delta = current_length - target_length
    shrink_start = delta // 2
    shrink_end = delta - shrink_start
    return (int(start) + shrink_start, int(end) - shrink_end)


def fit_axis_to_length_around_center(start, end, target_length, max_length):
    target_length = max(1, min(int(target_length), int(max_length)))
    center = (float(start) + float(end)) / 2.0
    fitted_start = int(round(center - (target_length / 2.0)))
    fitted_end = fitted_start + target_length
    return shift_axis_into_image(fitted_start, fitted_end, max_length)


def fit_box_to_ratio_inside_image(box, ratio_width, ratio_height, image_width, image_height):
    if box is None or ratio_width is None or ratio_height is None:
        return box

    fit_width, fit_height = get_max_in_image_size_for_ratio(
        ratio_width,
        ratio_height,
        image_width,
        image_height,
    )
    fitted_left, fitted_right = fit_axis_to_length_around_center(
        box[0],
        box[2],
        fit_width,
        image_width,
    )
    fitted_top, fitted_bottom = fit_axis_to_length_around_center(
        box[1],
        box[3],
        fit_height,
        image_height,
    )
    return (fitted_left, fitted_top, fitted_right, fitted_bottom)


def get_ratio_fit_crop_loss_percent(box, ratio_width, ratio_height, image_width, image_height):
    if box is None or ratio_width is None or ratio_height is None:
        return 0.0

    current_width, current_height = get_box_dimensions(box)
    fit_width, fit_height = get_max_in_image_size_for_ratio(
        ratio_width,
        ratio_height,
        image_width,
        image_height,
    )
    width_loss = max(0.0, ((current_width - fit_width) / max(1, current_width)) * 100.0)
    height_loss = max(0.0, ((current_height - fit_height) / max(1, current_height)) * 100.0)
    return max(width_loss, height_loss)


def should_crop_ratio_overflow(
    box,
    ratio_width,
    ratio_height,
    image_width,
    image_height,
    overflow_mode="auto",
    auto_crop_max_loss_percent=20,
    min_target_width=None,
    min_target_height=None,
):
    normalized_mode = str(overflow_mode).strip().lower()
    if normalized_mode == "crop":
        return True
    if normalized_mode == "pad":
        return False

    try:
        max_loss_percent = max(0.0, float(auto_crop_max_loss_percent))
    except (TypeError, ValueError):
        max_loss_percent = 20.0

    fit_width, fit_height = get_max_in_image_size_for_ratio(
        ratio_width,
        ratio_height,
        image_width,
        image_height,
    )
    if min_target_width is not None and fit_width < int(min_target_width):
        return False
    if min_target_height is not None and fit_height < int(min_target_height):
        return False

    crop_loss_percent = get_ratio_fit_crop_loss_percent(
        box,
        ratio_width,
        ratio_height,
        image_width,
        image_height,
    )
    return crop_loss_percent <= max_loss_percent


def adjust_box_to_aspect_ratio(box, ratio_width, ratio_height, image_width, image_height):
    if box is None or ratio_width is None or ratio_height is None:
        return box

    left, top, right, bottom = box
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    current_ratio_scaled = box_width * ratio_height
    target_ratio_scaled = box_height * ratio_width

    if current_ratio_scaled == target_ratio_scaled:
        return (int(left), int(top), int(right), int(bottom))

    if current_ratio_scaled < target_ratio_scaled:
        target_width = (box_height * ratio_width + ratio_height - 1) // ratio_height
        expanded_left, expanded_right = expand_axis_to_length(left, right, target_width, image_width)
        return (expanded_left, int(top), expanded_right, int(bottom))

    target_height = (box_width * ratio_height + ratio_width - 1) // ratio_width
    expanded_top, expanded_bottom = expand_axis_to_length(top, bottom, target_height, image_height)
    return (int(left), expanded_top, int(right), expanded_bottom)


def adjust_box_to_aspect_ratio_with_expand_compensation(
    box,
    ratio_width,
    ratio_height,
    image_width,
    image_height,
    original_box=None,
):
    if box is None or ratio_width is None or ratio_height is None:
        return box

    left, top, right, bottom = box
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    original_width, original_height = get_box_dimensions(original_box)
    current_ratio_scaled = box_width * ratio_height
    target_ratio_scaled = box_height * ratio_width

    if current_ratio_scaled == target_ratio_scaled:
        return (int(left), int(top), int(right), int(bottom))

    if current_ratio_scaled < target_ratio_scaled:
        target_width = (box_height * ratio_width + ratio_height - 1) // ratio_height
        if target_width <= image_width:
            return adjust_box_to_aspect_ratio(box, ratio_width, ratio_height, image_width, image_height)

        expanded_left, expanded_right = expand_axis_to_length(left, right, image_width, image_width)
        expanded_width = max(1, expanded_right - expanded_left)
        compensated_height = max(1, (expanded_width * ratio_height) // ratio_width)
        shrunk_top, shrunk_bottom = shrink_axis_to_length_with_min(
            top,
            bottom,
            compensated_height,
            original_height,
        )
        corrected_height = max(1, shrunk_bottom - shrunk_top)
        if expanded_width * ratio_height >= corrected_height * ratio_width:
            return (expanded_left, shrunk_top, expanded_right, shrunk_bottom)

        target_width = (corrected_height * ratio_width + ratio_height - 1) // ratio_height
        padded_left, padded_right = expand_axis_to_length_unbounded(
            expanded_left,
            expanded_right,
            target_width,
        )
        return (padded_left, shrunk_top, padded_right, shrunk_bottom)

    target_height = (box_width * ratio_height + ratio_width - 1) // ratio_width
    if target_height <= image_height:
        return adjust_box_to_aspect_ratio(box, ratio_width, ratio_height, image_width, image_height)

    expanded_top, expanded_bottom = expand_axis_to_length(top, bottom, image_height, image_height)
    expanded_height = max(1, expanded_bottom - expanded_top)
    compensated_width = max(1, (expanded_height * ratio_width) // ratio_height)
    shrunk_left, shrunk_right = shrink_axis_to_length_with_min(
        left,
        right,
        compensated_width,
        original_width,
    )
    corrected_width = max(1, shrunk_right - shrunk_left)
    if corrected_width * ratio_height <= expanded_height * ratio_width:
        return (shrunk_left, expanded_top, shrunk_right, expanded_bottom)

    target_height = (corrected_width * ratio_height + ratio_width - 1) // ratio_width
    padded_top, padded_bottom = expand_axis_to_length_unbounded(
        expanded_top,
        expanded_bottom,
        target_height,
    )
    return (shrunk_left, padded_top, shrunk_right, padded_bottom)


def apply_aspect_ratio_spec_to_box(
    box,
    aspect_spec,
    image_width,
    image_height,
    expand_pixels=0,
    original_box=None,
    overflow_mode="auto",
    auto_crop_max_loss_percent=20,
):
    if box is None or aspect_spec is None or aspect_spec["kind"] == "none":
        return box

    ratio_width = aspect_spec["ratio_width"]
    ratio_height = aspect_spec["ratio_height"]
    if would_ratio_correction_overflow_image(box, ratio_width, ratio_height, image_width, image_height):
        min_target_width = aspect_spec.get("resize_width")
        min_target_height = aspect_spec.get("resize_height")
        if should_crop_ratio_overflow(
            box,
            ratio_width,
            ratio_height,
            image_width,
            image_height,
            overflow_mode=overflow_mode,
            auto_crop_max_loss_percent=auto_crop_max_loss_percent,
            min_target_width=min_target_width,
            min_target_height=min_target_height,
        ):
            return fit_box_to_ratio_inside_image(
                box,
                ratio_width,
                ratio_height,
                image_width,
                image_height,
            )

    if normalize_expand_pixels(expand_pixels) > 0:
        return adjust_box_to_aspect_ratio_with_expand_compensation(
            box,
            ratio_width,
            ratio_height,
            image_width,
            image_height,
            original_box=original_box,
        )

    return adjust_box_to_aspect_ratio(
        box,
        ratio_width,
        ratio_height,
        image_width,
        image_height,
    )


def crop_with_padding(source_image, box):
    if box is None:
        return None, False

    left, top, right, bottom = [int(round(value)) for value in box]
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    padded = Image.new("RGBA", (crop_width, crop_height), (255, 255, 255, 255))

    image_width, image_height = source_image.size
    used_padding = left < 0 or top < 0 or right > image_width or bottom > image_height
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image_width, right)
    source_bottom = min(image_height, bottom)

    if source_right > source_left and source_bottom > source_top:
        visible_region = source_image.crop((source_left, source_top, source_right, source_bottom)).convert("RGBA")
        padded.paste(visible_region, (source_left - left, source_top - top))

    return padded, used_padding


def resize_image_to_target(image, target_width, target_height):
    if image is None or target_width is None or target_height is None:
        return image

    if image.size == (target_width, target_height):
        return image

    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def maybe_upscale_small_output(image, aspect_spec, allow_upscale):
    if image is None or not allow_upscale or aspect_spec is None or aspect_spec["kind"] != "size":
        return image, False

    target_width = aspect_spec["resize_width"]
    target_height = aspect_spec["resize_height"]
    if target_width is None or target_height is None:
        return image, False

    image_width, image_height = image.size
    if image_width >= target_width and image_height >= target_height:
        return image, False

    return resize_image_to_target(image, target_width, target_height), True


def expand_box_to_size(box, target_width, target_height, image_width, image_height):
    if box is None:
        return None

    left, top, right, bottom = box
    expanded_left, expanded_right = expand_axis_to_length(left, right, target_width, image_width)
    expanded_top, expanded_bottom = expand_axis_to_length(top, bottom, target_height, image_height)
    return shift_box_into_image((expanded_left, expanded_top, expanded_right, expanded_bottom), image_width, image_height)


def expand_box_to_size_within_image_limit(box, target_width, target_height, image_width, image_height):
    if box is None:
        return None, False, False

    current_width, current_height = get_box_dimensions(box)
    if target_width <= image_width and target_height <= image_height:
        desired_width = target_width
        desired_height = target_height
        reached_target = True
    else:
        desired_width, desired_height = get_max_in_image_size_for_ratio(
            target_width,
            target_height,
            image_width,
            image_height,
        )
        reached_target = desired_width >= target_width and desired_height >= target_height

    desired_width = max(current_width, int(desired_width))
    desired_height = max(current_height, int(desired_height))
    if desired_width == current_width and desired_height == current_height:
        return box, reached_target, False

    expanded_box = expand_box_to_size(box, desired_width, desired_height, image_width, image_height)
    return expanded_box, reached_target, True


def get_max_in_image_size_for_ratio(ratio_width, ratio_height, image_width, image_height):
    if ratio_width is None or ratio_height is None or image_width <= 0 or image_height <= 0:
        return (max(1, image_width), max(1, image_height))

    if image_width * ratio_height <= image_height * ratio_width:
        fit_width = image_width
        fit_height = max(1, (image_width * ratio_height) // ratio_width)
        return (fit_width, fit_height)

    fit_height = image_height
    fit_width = max(1, (image_height * ratio_width) // ratio_height)
    return (fit_width, fit_height)


def format_ratio_delta(current_size, target_size):
    if target_size <= 0:
        return "0.00%"
    return f"{((target_size - current_size) / target_size) * 100:.2f}%"


def append_crop_issue_log(
    log_entries,
    source_name,
    box_name,
    aspect_spec,
    statuses,
    current_width=None,
    current_height=None,
):
    if not statuses:
        return

    log_entries.append(
        {
            "source": source_name,
            "box": box_name,
            "spec": aspect_spec["text"] if aspect_spec is not None else "",
            "current_size": f"{current_width}x{current_height}" if current_width is not None and current_height is not None else "",
            "status": ",".join(statuses),
        }
    )


def apply_size_correction_to_box(
    box,
    aspect_spec,
    image_width,
    image_height,
    expand_pixels=0,
    size_expand_threshold_percent=20,
):
    if box is None or aspect_spec is None or aspect_spec["kind"] != "size":
        return box, None, None

    target_width = aspect_spec["resize_width"]
    target_height = aspect_spec["resize_height"]
    if target_width is None or target_height is None:
        return box, None, None

    current_width = max(1, int(round(box[2] - box[0])))
    current_height = max(1, int(round(box[3] - box[1])))
    size_info = {
        "current_width": current_width,
        "current_height": current_height,
        "target_width": target_width,
        "target_height": target_height,
        "action": "already_target",
    }

    if current_width > target_width or current_height > target_height:
        size_info["action"] = "scaled_down_to_target"
        return box, (target_width, target_height), size_info

    if current_width < target_width or current_height < target_height:
        box, reached_target_in_image, expanded_in_image = expand_box_to_size_within_image_limit(
            box,
            target_width,
            target_height,
            image_width,
            image_height,
        )
        current_width = max(1, int(round(box[2] - box[0])))
        current_height = max(1, int(round(box[3] - box[1])))
        if expanded_in_image and reached_target_in_image:
            size_info["action"] = "expanded_to_target"
            return box, None, size_info

        scale_up_factor = max(target_width / current_width, target_height / current_height)
        max_scale_up_factor = 1.0 + (max(0.0, float(size_expand_threshold_percent)) / 100.0)
        if scale_up_factor <= max_scale_up_factor:
            if (
                normalize_expand_pixels(expand_pixels) > 0
                and (target_width > image_width or target_height > image_height)
            ):
                limited_width, limited_height = get_max_in_image_size_for_ratio(
                    target_width,
                    target_height,
                    image_width,
                    image_height,
                )
                size_info["action"] = "expanded_to_image_limit"
                expanded_box = expand_box_to_size(
                    box,
                    limited_width,
                    limited_height,
                    image_width,
                    image_height,
                )
                return expanded_box, (target_width, target_height), size_info

            size_info["action"] = "expanded_to_target"
            expanded_box = expand_box_to_size(box, target_width, target_height, image_width, image_height)
            return expanded_box, None, size_info

        size_info["action"] = "skipped_over_expand_threshold"
        return box, None, size_info

    return box, None, size_info
