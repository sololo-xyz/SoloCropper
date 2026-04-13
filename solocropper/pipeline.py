# SoloCropper
# Copyright (c) 2026 Solo
# Original work by Solo | https://sololo.xyz

import shutil
from pathlib import Path

from PIL import Image

from . import config, detection
from .console import print_error, print_notice
from .geometry import expand_box_by_pixels
from .rendering import (
    build_save_config_summary,
    save_box_variants,
    save_region_crops,
    write_size_fix_log,
)

GENERIC_OBJECT_FALLBACK_SEG_CONF_THRES = 0.05


def clear_directory_contents(target_dir):
    if not target_dir.exists():
        return

    for child in target_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def main(base_dir=None):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path(base_dir)

    config_path = base_dir / config.CONFIG_PATH
    loaded_config = config.load_config(config_path)

    if loaded_config is None:
        return

    config.apply_runtime_config(loaded_config)

    output_format = loaded_config["output_format"]
    box_mode = loaded_config["box_mode"]
    crop_mode = loaded_config["crop_mode"]
    device = loaded_config["device"]
    if device.startswith("mps"):
        print_error(
            'Error: device="mps" is not supported in this project. '
            'Use device="cpu", device="auto", or a CUDA device such as "cuda:0".'
        )
        return
    models_config = loaded_config["models"]
    paths_config = loaded_config["paths"]
    thresholds_config = loaded_config["thresholds"]
    seg_model_file = config.resolve_config_path(base_dir, models_config["segmentation"])
    pose_model_file = config.resolve_config_path(base_dir, models_config["pose"])
    input_dir = config.resolve_config_path(base_dir, paths_config["input_dir"])
    output_dir = config.resolve_config_path(base_dir, paths_config["output_dir"])
    clear_output_dir_on_start = paths_config["clear_output_dir_on_start"]
    seg_conf_threshold = thresholds_config["segmentation"]
    pose_conf_threshold = thresholds_config["pose"]
    seg_fallback_conf_threshold = thresholds_config["segmentation_fallback"]
    auto_crop_max_loss_percent = loaded_config["aspect_ratio_overflow"]["auto_crop_max_loss_percent"]
    size_expand_threshold_percent = loaded_config["size_expand_threshold_percent"]
    upscale_small_outputs = loaded_config["upscale_small_outputs"]
    box_settings = loaded_config["box_settings"]

    if not box_mode and not crop_mode:
        print_notice("Notice: box_mode and crop_mode are both disabled. Nothing to output.")
        return

    enabled_box_names = [
        box_name for box_name in config.BOX_SETTING_KEYS if config.is_box_enabled(box_settings, box_name)
    ]
    if not enabled_box_names:
        print_notice("Notice: all boxes are disabled in box_settings. Nothing to process.")
        return
    box_spec_map = {
        box_name: config.get_box_aspect_ratio_specs(box_settings, box_name) for box_name in enabled_box_names
    }

    if not seg_model_file.exists():
        print_error(f"Error: segmentation model file not found: {seg_model_file}")
        return

    if not pose_model_file.exists():
        print_error(f"Error: pose model file not found: {pose_model_file}")
        return

    if not input_dir.exists():
        print_error(f"Error: input directory not found: {input_dir}")
        return

    if output_dir.exists() and not output_dir.is_dir():
        print_error(f"Error: output path exists but is not a directory: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_output_dir_on_start:
        clear_directory_contents(output_dir)
        print_notice(f"Cleared output directory: {output_dir}")

    images = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in config.IMAGE_EXTENSIONS)
    if not images:
        print_notice("No supported images found in the input directory.")
        return

    size_fix_log_entries = []
    completed_count = 0
    skipped_count = 0
    failed_count = 0

    print(
        f"Output config: format={output_format}, "
        f"{build_save_config_summary(output_format)}, "
        f"box_mode={box_mode}, crop_mode={crop_mode}, "
        f"device={device}, "
        f"seg_conf_threshold={seg_conf_threshold:g}, "
        f"pose_conf_threshold={pose_conf_threshold:g}, "
        f"seg_fallback_conf_threshold={seg_fallback_conf_threshold:g}, "
        f"keypoint_conf_threshold={config.KEYPOINT_CONF_THRES:g}, "
        f"auto_crop_max_loss_percent={auto_crop_max_loss_percent:g}, "
        f"size_expand_threshold_percent={size_expand_threshold_percent:g}, "
        f"upscale_small_outputs={upscale_small_outputs}, "
        f"enabled_boxes={','.join(enabled_box_names)}"
    )
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Loading segmentation model: {seg_model_file}")
    seg_model = detection.YOLO(str(seg_model_file))
    print(f"Loading pose model: {pose_model_file}")
    pose_model = detection.YOLO(str(pose_model_file))

    print(f"Processing {len(images)} images...")
    for index, image_path in enumerate(images, start=1):
        try:
            with Image.open(image_path) as source_image:
                model_input_image = detection.get_model_input_image(source_image)
                generic_object_fallback_used = False
                generic_object_count = 0

                seg_result = detection.predict_person_result(
                    seg_model,
                    model_input_image,
                    seg_conf_threshold,
                    device,
                    retina_masks=True,
                )
                main_person, person_count = detection.choose_main_person(seg_result)
                recovered_with_seg_fallback = False

                if main_person is None and seg_fallback_conf_threshold < seg_conf_threshold:
                    fallback_seg_result = detection.predict_person_result(
                        seg_model,
                        model_input_image,
                        seg_fallback_conf_threshold,
                        device,
                        retina_masks=True,
                    )
                    fallback_main_person, fallback_person_count = detection.choose_main_person(fallback_seg_result)
                    if fallback_main_person is not None:
                        seg_result = fallback_seg_result
                        main_person = fallback_main_person
                        person_count = fallback_person_count
                        recovered_with_seg_fallback = True

                pose_result = detection.predict_person_result(
                    pose_model,
                    model_input_image,
                    pose_conf_threshold,
                    device,
                )

                seg_person_index = None
                pose_person_index = None
                if main_person is not None:
                    box = main_person["box"]
                    seg_person_index = main_person["index"]
                    if recovered_with_seg_fallback:
                        print_notice(
                            f"[{index}/{len(images)}] Recovered {image_path.name} "
                            f"with lower segmentation threshold ({seg_fallback_conf_threshold:g})"
                        )
                    pose_person_index = detection.choose_pose_person(pose_result, box)
                else:
                    pose_main_person, pose_person_count = detection.choose_main_person(pose_result)
                    if pose_main_person is None:
                        generic_seg_conf_threshold = min(
                            seg_fallback_conf_threshold,
                            GENERIC_OBJECT_FALLBACK_SEG_CONF_THRES,
                        )
                        generic_seg_result = detection.predict_detection_result(
                            seg_model,
                            model_input_image,
                            generic_seg_conf_threshold,
                            device,
                            retina_masks=True,
                        )
                        generic_main_object, generic_object_count = detection.choose_main_object(
                            generic_seg_result
                        )
                        if generic_main_object is None:
                            skipped_count += 1
                            print_error(
                                f"[{index}/{len(images)}] Skipped {image_path.name}: "
                                "no person detected"
                            )
                            continue

                        box = generic_main_object["box"]
                        person_count = 0
                        seg_result = generic_seg_result
                        generic_object_fallback_used = True
                        print_notice(
                            f"[{index}/{len(images)}] Using generic object fallback for "
                            f"{image_path.name}: no human detected, saving full box only"
                        )
                    else:
                        box = pose_main_person["box"]
                        person_count = pose_person_count
                        pose_person_index = pose_main_person["index"]
                        print_notice(
                            f"[{index}/{len(images)}] Using pose fallback for {image_path.name}: "
                            "segmentation found no person"
                        )

                pose_type = detection.classify_pose_type(
                    pose_result,
                    pose_person_index,
                    seg_result=seg_result,
                    seg_person_index=seg_person_index,
                    box=box,
                )
                lines = detection.collect_horizontal_lines(
                    pose_result,
                    pose_person_index,
                    box,
                    pose_type,
                    seg_result=seg_result,
                    seg_person_index=seg_person_index,
                )

                image_width, image_height = source_image.size
                region_items = detection.build_region_items(
                    lines,
                    box,
                    image_width,
                    image_height,
                    seg_result=seg_result,
                    seg_person_index=seg_person_index,
                    pose_result=pose_result,
                    pose_person_index=pose_person_index,
                )
                enabled_region_items = detection.filter_region_items(region_items, box_settings)
                expanded_region_items = []
                for item in enabled_region_items:
                    expand_pixels = config.get_box_expand(box_settings, item["label"])
                    expanded_item = item.copy()
                    expanded_item["original_box"] = item["box"]
                    expanded_item["box"] = expand_box_by_pixels(
                        item["box"],
                        expand_pixels,
                        image_width,
                        image_height,
                    )
                    expanded_item["expand_pixels"] = expand_pixels
                    expanded_item["overflow_mode"] = config.get_box_overflow_mode(box_settings, item["label"])
                    expanded_region_items.append(expanded_item)

                full_box = None
                full_original_box = None
                full_expand_pixels = 0
                full_overflow_mode = config.get_box_overflow_mode(box_settings, "full")
                if generic_object_fallback_used or config.is_box_enabled(box_settings, "full"):
                    full_original_box = box
                    full_expand_pixels = config.get_box_expand(box_settings, "full")
                    full_box = expand_box_by_pixels(
                        box,
                        full_expand_pixels,
                        image_width,
                        image_height,
                    )

                label_anchor_box = full_box if full_box is not None else box

                saved_outputs = []
                if box_mode and (full_box is not None or expanded_region_items):
                    box_count = save_box_variants(
                        source_image,
                        output_dir,
                        image_path,
                        label_anchor_box,
                        full_box,
                        full_original_box,
                        full_expand_pixels,
                        full_overflow_mode,
                        expanded_region_items,
                        output_format,
                        box_spec_map,
                        auto_crop_max_loss_percent,
                    )
                    saved_outputs.append(f"box_images={box_count}")

                if crop_mode:
                    crop_count = save_region_crops(
                        source_image,
                        output_dir,
                        image_path,
                        full_box,
                        full_original_box,
                        full_expand_pixels,
                        full_overflow_mode,
                        expanded_region_items,
                        output_format,
                        box_spec_map,
                        size_fix_log_entries,
                        auto_crop_max_loss_percent,
                        size_expand_threshold_percent,
                        upscale_small_outputs,
                    )
                    saved_outputs.append(f"crops={crop_count}")

            completed_count += 1
            output_summary = ", ".join(saved_outputs) if saved_outputs else "no outputs"
            generic_object_summary = (
                f", generic_objects={generic_object_count}" if generic_object_fallback_used else ""
            )
            print(
                f"[{index}/{len(images)}] Done {image_path.name}: "
                f"detected_people={person_count}{generic_object_summary}, "
                f"enabled_region_boxes={len(expanded_region_items)}, "
                f"{output_summary}"
            )
        except detection.DeviceConfigurationError as error:
            failed_count += 1
            print_error(f"[{index}/{len(images)}] Stopped on {image_path.name}: {error}")
            break
        except Exception as error:
            failed_count += 1
            print_error(f"[{index}/{len(images)}] Failed {image_path.name}: {error}")

    write_size_fix_log(output_dir, size_fix_log_entries)
    if size_fix_log_entries:
        print_notice(f"Size fix log written: {config.SIZE_FIX_LOG_FILENAME} ({len(size_fix_log_entries)} entries)")

    print(
        "Run summary: "
        f"completed={completed_count}, skipped={skipped_count}, failed={failed_count}"
    )
    print("Done.")
