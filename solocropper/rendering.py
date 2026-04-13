# SoloCropper
# Copyright (c) 2026 Solo
# Original work by Solo | https://sololo.xyz

from PIL import Image, ImageDraw, ImageFont

from . import config
from .geometry import (
    append_crop_issue_log,
    apply_aspect_ratio_spec_to_box,
    apply_size_correction_to_box,
    crop_with_padding,
    maybe_upscale_small_output,
    resize_image_to_target,
)


def load_label_font():
    for font_path in config.FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, config.LABEL_FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def prepare_image_for_saving(image, output_path):
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".bmp"}:
        return image.convert("RGB")
    return image


def get_save_kwargs(output_path):
    suffix = output_path.suffix.lower()
    if suffix == ".png":
        return {"compress_level": config.PNG_COMPRESS_LEVEL}
    if suffix in {".jpg", ".jpeg"}:
        return {"quality": config.JPEG_QUALITY}
    if suffix == ".webp":
        return {"quality": config.WEBP_QUALITY, "method": config.WEBP_METHOD}
    return {}


def save_image(image, output_path):
    image_to_save = prepare_image_for_saving(image, output_path)
    image_to_save.save(output_path, **get_save_kwargs(output_path))


def build_save_config_summary(output_format):
    if output_format == "png":
        return f"png_compress_level={config.PNG_COMPRESS_LEVEL}"
    if output_format in {"jpg", "jpeg"}:
        return f"jpeg_quality={config.JPEG_QUALITY}"
    if output_format == "webp":
        return f"webp_quality={config.WEBP_QUALITY}, webp_method={config.WEBP_METHOD}"
    return "save_settings=default"


def build_output_path(output_dir, source_path, suffix_name, output_format):
    return output_dir / f"{source_path.stem}_{suffix_name}.{output_format}"


def write_size_fix_log(output_dir, log_entries):
    log_path = output_dir / config.SIZE_FIX_LOG_FILENAME

    if not log_entries:
        if log_path.exists():
            log_path.unlink()
        return

    lines = []
    for entry in log_entries:
        lines.append(
            " | ".join(
                [
                    entry["source"],
                    entry["box"],
                    f"spec={entry['spec']}",
                    f"current={entry['current_size']}",
                    entry["status"],
                ]
            )
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_flagged_crop_output_dir(output_dir):
    flagged_dir = output_dir / config.FLAGGED_CROP_SUBDIR
    flagged_dir.mkdir(parents=True, exist_ok=True)
    return flagged_dir


def make_region_item_variant(region_item, variant_box):
    variant_item = region_item.copy()
    variant_item["box"] = variant_box
    return variant_item


def shift_box(box, offset_x, offset_y):
    if box is None:
        return None

    left, top, right, bottom = [int(round(value)) for value in box]
    return (
        left + int(offset_x),
        top + int(offset_y),
        right + int(offset_x),
        bottom + int(offset_y),
    )


def shift_region_item(region_item, offset_x, offset_y):
    shifted_item = region_item.copy()
    shifted_item["box"] = shift_box(region_item.get("box"), offset_x, offset_y)
    if region_item.get("line_y") is not None:
        shifted_item["line_y"] = int(round(region_item["line_y"])) + int(offset_y)
    return shifted_item


def build_annotation_canvas(source_image, label_anchor_box, region_items, main_box=None):
    source_rgba = source_image.convert("RGBA")
    image_width, image_height = source_rgba.size
    min_left = 0
    min_top = 0
    max_right = image_width
    max_bottom = image_height

    all_boxes = []
    if main_box is not None:
        all_boxes.append(main_box)
    if label_anchor_box is not None:
        all_boxes.append(label_anchor_box)
    for item in region_items:
        item_box = item.get("box")
        if item_box is not None:
            all_boxes.append(item_box)

    for box in all_boxes:
        left, top, right, bottom = [int(round(value)) for value in box]
        min_left = min(min_left, left)
        min_top = min(min_top, top)
        max_right = max(max_right, right)
        max_bottom = max(max_bottom, bottom)

    if min_left >= 0 and min_top >= 0 and max_right <= image_width and max_bottom <= image_height:
        return source_rgba, label_anchor_box, region_items, main_box

    offset_x = -min_left
    offset_y = -min_top
    expanded_canvas = Image.new(
        "RGBA",
        (max_right - min_left, max_bottom - min_top),
        (255, 255, 255, 255),
    )
    expanded_canvas.paste(source_rgba, (offset_x, offset_y))

    shifted_label_anchor_box = shift_box(label_anchor_box, offset_x, offset_y)
    shifted_region_items = [
        shift_region_item(item, offset_x, offset_y) for item in region_items
    ]
    shifted_main_box = shift_box(main_box, offset_x, offset_y)
    return expanded_canvas, shifted_label_anchor_box, shifted_region_items, shifted_main_box


def boxes_overlap(box_a, box_b, padding=2):
    return not (
        box_a[2] + padding <= box_b[0]
        or box_b[2] + padding <= box_a[0]
        or box_a[3] + padding <= box_b[1]
        or box_b[3] + padding <= box_a[1]
    )


def clamp_label_position(value, max_value):
    return max(0, min(int(round(value)), max(0, int(max_value))))


def build_label_y_candidates(preferred_label_y, label_height, image_height):
    max_label_y = max(0, image_height - label_height - 1)
    clamped_preferred = clamp_label_position(preferred_label_y, max_label_y)
    candidates = []
    seen = set()
    step = max(1, label_height + 2)

    for distance in range(0, max_label_y + step, step):
        raw_values = [clamped_preferred] if distance == 0 else [clamped_preferred + distance, clamped_preferred - distance]
        for raw_value in raw_values:
            candidate = clamp_label_position(raw_value, max_label_y)
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    return candidates if candidates else [0]


def choose_label_position(
    preferred_label_x,
    alternate_label_x,
    preferred_label_y,
    label_width,
    label_height,
    image_width,
    image_height,
    occupied_boxes,
):
    max_label_x = max(0, image_width - label_width - 1)
    clamped_preferred_y = clamp_label_position(preferred_label_y, max(0, image_height - label_height - 1))
    candidate_xs = []
    for raw_x in (preferred_label_x, alternate_label_x):
        candidate_x = clamp_label_position(raw_x, max_label_x)
        if candidate_x not in candidate_xs:
            candidate_xs.append(candidate_x)

    candidate_ys = build_label_y_candidates(preferred_label_y, label_height, image_height)
    best_box = None
    best_score = None

    for x_rank, candidate_x in enumerate(candidate_xs):
        for candidate_y in candidate_ys:
            candidate_box = (
                candidate_x,
                candidate_y,
                candidate_x + label_width,
                candidate_y + label_height,
            )
            overlap_count = sum(1 for occupied_box in occupied_boxes if boxes_overlap(candidate_box, occupied_box))
            distance = abs(candidate_y - clamped_preferred_y)
            score = (overlap_count, distance, x_rank)
            if best_score is None or score < best_score:
                best_score = score
                best_box = candidate_box
            if overlap_count == 0:
                return candidate_box

    return best_box


def draw_annotations(source_image, output_path, label_anchor_box, region_items, main_box=None):
    annotated, label_anchor_box, region_items, main_box = build_annotation_canvas(
        source_image,
        label_anchor_box,
        region_items,
        main_box=main_box,
    )
    draw = ImageDraw.Draw(annotated)
    font = load_label_font()
    image_width, image_height = annotated.size

    if main_box is not None:
        draw.rectangle(main_box, outline=config.MAIN_BOX_COLOR, width=config.LINE_WIDTH)

    if label_anchor_box is None:
        label_anchor_box = main_box if main_box is not None else (0, 0, image_width, image_height)

    for item in region_items:
        line_box = item["box"]
        if line_box is None:
            continue
        draw.rectangle(line_box, outline=item["color"], width=config.LINE_BOX_WIDTH)

    occupied_label_boxes = []
    for item in region_items:
        label = item["label"]
        line_y = item["line_y"]
        color = item["color"]
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_width = text_width + (config.LABEL_PADDING_X * 2)
        label_height = text_height + (config.LABEL_PADDING_Y * 2)
        left_space = max(0, label_anchor_box[0])
        right_space = max(0, image_width - label_anchor_box[2])

        if right_space >= left_space:
            preferred_label_x = label_anchor_box[2] + config.LABEL_GAP
            alternate_label_x = label_anchor_box[0] - config.LABEL_GAP - label_width
        else:
            preferred_label_x = label_anchor_box[0] - config.LABEL_GAP - label_width
            alternate_label_x = label_anchor_box[2] + config.LABEL_GAP
        preferred_label_y = int(round(line_y - (label_height / 2.0)))
        background_box = choose_label_position(
            preferred_label_x,
            alternate_label_x,
            preferred_label_y,
            label_width,
            label_height,
            image_width,
            image_height,
            occupied_label_boxes,
        )
        label_x = background_box[0]
        label_y = background_box[1]
        draw.rectangle(background_box, fill=config.LABEL_BG_COLOR)
        draw.text(
            (
                label_x + config.LABEL_PADDING_X - text_bbox[0],
                label_y + config.LABEL_PADDING_Y - text_bbox[1],
            ),
            label,
            fill=color,
            font=font,
        )
        occupied_label_boxes.append(background_box)

    save_image(annotated, output_path)


def save_box_variants(
    source_image,
    output_dir,
    source_path,
    label_anchor_box,
    full_box,
    full_original_box,
    full_expand_pixels,
    full_overflow_mode,
    region_items,
    output_format,
    box_spec_map,
    auto_crop_max_loss_percent,
):
    image_width, image_height = source_image.size
    saved_count = 0

    if full_box is not None or region_items:
        output_path = build_output_path(output_dir, source_path, "box_0", output_format)
        draw_annotations(
            source_image,
            output_path,
            label_anchor_box,
            region_items,
            main_box=full_box,
        )
        saved_count += 1

    corrected_main_box = None
    has_correction = False
    if full_box is not None:
        full_aspect_spec = config.get_preview_aspect_ratio_spec(box_spec_map, "full")
        corrected_main_box = apply_aspect_ratio_spec_to_box(
            full_box,
            full_aspect_spec,
            image_width,
            image_height,
            expand_pixels=full_expand_pixels,
            original_box=full_original_box,
            overflow_mode=full_overflow_mode,
            auto_crop_max_loss_percent=auto_crop_max_loss_percent,
        )
        if full_aspect_spec["kind"] != "none":
            has_correction = True

    corrected_region_items = []
    for item in region_items:
        aspect_spec = config.get_preview_aspect_ratio_spec(box_spec_map, item["label"])
        corrected_box = apply_aspect_ratio_spec_to_box(
            item["box"],
            aspect_spec,
            image_width,
            image_height,
            expand_pixels=item.get("expand_pixels", 0),
            original_box=item.get("original_box"),
            overflow_mode=item.get("overflow_mode", config.OVERFLOW_MODE_AUTO),
            auto_crop_max_loss_percent=auto_crop_max_loss_percent,
        )
        if corrected_box is None:
            continue
        corrected_region_items.append(make_region_item_variant(item, corrected_box))
        if aspect_spec["kind"] != "none":
            has_correction = True

    if has_correction and (corrected_main_box is not None or corrected_region_items):
        corrected_anchor_box = corrected_main_box if corrected_main_box is not None else label_anchor_box
        output_path = build_output_path(output_dir, source_path, "box_1", output_format)
        draw_annotations(
            source_image,
            output_path,
            corrected_anchor_box,
            corrected_region_items,
            main_box=corrected_main_box,
        )
        saved_count += 1

    return saved_count


def save_region_crops(
    source_image,
    output_dir,
    source_path,
    full_box,
    full_original_box,
    full_expand_pixels,
    full_overflow_mode,
    region_items,
    output_format,
    box_spec_map,
    size_fix_log_entries,
    auto_crop_max_loss_percent,
    size_expand_threshold_percent,
    upscale_small_outputs,
):
    image_width, image_height = source_image.size
    saved_count = 0

    if full_box is not None:
        for aspect_spec in box_spec_map.get("full", []):
            adjusted_full_box = apply_aspect_ratio_spec_to_box(
                full_box,
                aspect_spec,
                image_width,
                image_height,
                expand_pixels=full_expand_pixels,
                original_box=full_original_box,
                overflow_mode=full_overflow_mode,
                auto_crop_max_loss_percent=auto_crop_max_loss_percent,
            )
            sized_full_box, resize_target, size_info = apply_size_correction_to_box(
                adjusted_full_box,
                aspect_spec,
                image_width,
                image_height,
                expand_pixels=full_expand_pixels,
                size_expand_threshold_percent=size_expand_threshold_percent,
            )
            cropped_full_image, used_padding = crop_with_padding(source_image, sized_full_box)
            if cropped_full_image is None:
                continue
            if resize_target is not None:
                cropped_full_image = resize_image_to_target(
                    cropped_full_image,
                    resize_target[0],
                    resize_target[1],
                )
            cropped_full_image, was_upscaled = maybe_upscale_small_output(
                cropped_full_image,
                aspect_spec,
                upscale_small_outputs,
            )

            issue_flags = []
            log_statuses = []
            if size_info is not None and size_info["action"] in {"expanded_to_target", "expanded_to_image_limit"}:
                log_statuses.append("expanded")
            if size_info is not None and size_info["action"] == "skipped_over_expand_threshold":
                issue_flags.append("skipped_over_expand_threshold")
                log_statuses.append("skipped")
            if used_padding:
                issue_flags.append("padded")
                log_statuses.append("padded")
            if was_upscaled:
                issue_flags.append("upscaled_small_output")
                log_statuses.append("upscaled")

            crop_output_dir = get_flagged_crop_output_dir(output_dir) if issue_flags else output_dir
            full_output_path = build_output_path(
                crop_output_dir,
                source_path,
                f"crop_00_full_{aspect_spec['suffix']}",
                output_format,
            )
            save_image(cropped_full_image, full_output_path)
            if log_statuses:
                append_crop_issue_log(
                    size_fix_log_entries,
                    source_path.name,
                    "full",
                    aspect_spec,
                    log_statuses,
                    current_width=size_info["current_width"] if size_info is not None else None,
                    current_height=size_info["current_height"] if size_info is not None else None,
                )
            saved_count += 1

    sortable_items = [item for item in region_items if item["box"] is not None]
    sortable_items.sort(key=lambda item: item["box"][3], reverse=True)

    for order_index, item in enumerate(sortable_items, start=1):
        for aspect_spec in box_spec_map.get(item["label"], []):
            adjusted_box = apply_aspect_ratio_spec_to_box(
                item["box"],
                aspect_spec,
                image_width,
                image_height,
                expand_pixels=item.get("expand_pixels", 0),
                original_box=item.get("original_box"),
                overflow_mode=item.get("overflow_mode", config.OVERFLOW_MODE_AUTO),
                auto_crop_max_loss_percent=auto_crop_max_loss_percent,
            )
            sized_box, resize_target, size_info = apply_size_correction_to_box(
                adjusted_box,
                aspect_spec,
                image_width,
                image_height,
                expand_pixels=item.get("expand_pixels", 0),
                size_expand_threshold_percent=size_expand_threshold_percent,
            )
            cropped_image, used_padding = crop_with_padding(source_image, sized_box)
            if cropped_image is None:
                continue
            if resize_target is not None:
                cropped_image = resize_image_to_target(
                    cropped_image,
                    resize_target[0],
                    resize_target[1],
                )
            cropped_image, was_upscaled = maybe_upscale_small_output(
                cropped_image,
                aspect_spec,
                upscale_small_outputs,
            )

            issue_flags = []
            log_statuses = []
            if size_info is not None and size_info["action"] in {"expanded_to_target", "expanded_to_image_limit"}:
                log_statuses.append("expanded")
            if size_info is not None and size_info["action"] == "skipped_over_expand_threshold":
                issue_flags.append("skipped_over_expand_threshold")
                log_statuses.append("skipped")
            if used_padding:
                issue_flags.append("padded")
                log_statuses.append("padded")
            if was_upscaled:
                issue_flags.append("upscaled_small_output")
                log_statuses.append("upscaled")

            crop_output_dir = get_flagged_crop_output_dir(output_dir) if issue_flags else output_dir
            crop_output_path = build_output_path(
                crop_output_dir,
                source_path,
                f"crop_{order_index:02d}_{item['label']}_{aspect_spec['suffix']}",
                output_format,
            )
            save_image(cropped_image, crop_output_path)
            if log_statuses:
                append_crop_issue_log(
                    size_fix_log_entries,
                    source_path.name,
                    item["label"],
                    aspect_spec,
                    log_statuses,
                    current_width=size_info["current_width"] if size_info is not None else None,
                    current_height=size_info["current_height"] if size_info is not None else None,
                )
            saved_count += 1

    return saved_count
