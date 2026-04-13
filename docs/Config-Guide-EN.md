# SoloCropper Configuration Guide

This document explains what each setting in `config.toml` does. The configuration is organized into 7 core modules.

The current recommended keys are `[save].output_format` and `[crop_logic]`. Older configs are still compatible with:

- top-level `output_format`
- top-level `size_expand_threshold_percent`
- top-level `upscale_small_outputs`
- legacy `[box_rules]`

## 1. Operation Modes

Controls the most basic runtime behavior.

- `device`: Runtime device. `"auto"` is recommended.
  You can also use `"cpu"` or a CUDA device string such as `"cuda:0"` or `"cuda:1"`, but whether it actually works depends on whether the corresponding GPU exists in the current environment and can be used normally by PyTorch.
  When set to `auto`, the program chooses the device automatically and will try to fall back to CPU in some CUDA failure scenarios.

- `box_mode`: When set to `true`, the program outputs preview images with colored boxes and labels, which is useful for checking placement and composition.

- `crop_mode`: When set to `true`, the program outputs the actual cropped images.

Note: if both `box_mode` and `crop_mode` are `false`, the program will not output any images.

## 2. Paths & Models

### File Paths `[paths]`

- `input_dir`: Put the images to be processed in this folder. Only the current directory level is scanned; subdirectories are not included.

- `output_dir`: Output folder. Preview images, crops, logs, and flagged results are all written here.

- `flagged_crop_subdir`: Subfolder for specially handled outputs, such as white-padded results, crops that exceeded the box expansion threshold, or crops that were finally upscaled.

- `size_fix_log_filename`: Log filename used to record size-correction details.

- `clear_output_dir_on_start`: When set to `true`, the program clears the output directory before processing starts. Use this carefully.

### Model Paths `[models]`

- `segmentation`: Path to the YOLO segmentation model. This should point to the exact model file.

- `pose`: Path to the YOLO pose model. This should point to the exact model file.

## 3. Output Format & Quality

### Output Image Format

- `output_format`: Supported output formats are `"png"`, `"jpg"`, `"jpeg"`, `"webp"`, and `"bmp"`.
  - `png`: Good for high-quality datasets and detailed inspection, but usually slower and larger.
  - `jpg` / `jpeg`: Smaller files, suitable for quick previewing.
  - `webp`: Usually offers a good balance between file size and image quality.
  - `bmp`: Mainly for compatibility. Usually very large.

### Quality Controls

The following parameters apply to the selected output format above.

- `output_format`: The canonical location for selecting the output format.

- `png_compress_level` (`0-9`): Lower values are faster and produce larger files.

- `jpeg_quality` (`1-100`): Higher values give better image quality and larger files.

- `webp_quality` (`1-100`): Higher values give better image quality and larger files.

- `webp_method` (`0-6`): WEBP encoder effort. Lower values are faster; higher values are slower but usually compress better.

## 4. Crop Logic

Describes how the final crop box is adjusted based on the configured target ratio, target size, and related factors, using either the person main box or the current box for each region.

### How Fixed Output Sizes Are Handled

When `aspect_ratio_fix` uses a fixed output size such as `"512,768"`:

- The program first adjusts the crop box to the target aspect ratio.
- If the crop box is larger than target, the output image is scaled down to the target size.
- If the crop box is smaller than target, the program first tries to expand the crop box.
- If the required box growth exceeds `size_expand_threshold_percent`, the program stops expanding and marks the result.
- If `upscale_small_outputs = true`, the still-too-small output image will finally be upscaled to the target size.

### Crop Box Expansion and Image Upscaling `[crop_logic]`

- `size_expand_threshold_percent`: The percentage limit for how much the crop box may be enlarged, by scale, in order to reach the configured output size.

- `upscale_small_outputs`: If the actual crop size is smaller than the configured output size, this decides whether the result may be stretched up to the target size.

### Crop Box Decision Strategy `[aspect_ratio_overflow]`

- `auto_crop_max_loss_percent`: When a corrected crop box would go outside the source image in order to satisfy the configured ratio or size, this value is used to decide whether the program should crop inward and lose some content, or allow the crop box to go beyond the image and fill the extra area with white padding.
  This value represents the maximum crop-loss percentage the program is willing to accept.
  Lower values make the program prefer preserving composition and using white padding.
  Higher values make the program prefer cropping inward to avoid white borders.

  This setting only takes effect when a crop box uses `overflow_mode = "auto"`.

## 5. Detection Thresholds

### Detection Sensitivity `[thresholds]`

- `segmentation`: Main segmentation threshold (`0.0-1.0`).
  Lower values make it easier to recover difficult images, but increase the risk of false positives.

- `segmentation_fallback`: Backup threshold used only when the main segmentation pass fails to find any person.
  It should generally be lower than `segmentation`.

- `pose`: Pose threshold.
  Higher values usually make keypoints and horizontal lines more stable.

- `keypoint`: Minimum confidence for accepting keypoints into shoulder, waist, hip, knee, and ankle logic.
  Lower values are more tolerant, but also more likely to introduce jitter and error.

## 6. Boxes

For each image, up to 8 crop box groups can be generated based on different body regions. Each box can then be configured with different output ratios and sizes to produce one or more final crop results.

### 8 Box Groups

- `full`: Full-body box.
- `shoulder`: From the top of the person down to the shoulder line.
- `shoulder_only`: A narrower shoulder-focused region that uses the shoulder points as its boundary and does not widen its horizontal range because of arms, clothing, or hair.
- `waist`: From the top of the person down to the waist line (estimated, not always accurate).
- `hip`: From the top of the person down to the hip line.
- `glute`: From the top of the person down to the glute line (estimated, not always accurate).
- `knee`: From the top of the person down to the knee line.
- `ankle`: From the top of the person down to the ankle line.

### Box Settings

Every `[box_settings.<region>]` block contains the following common settings:

- `enabled`: Whether previewing or cropping for that region is enabled.

- `expand`: Crop box expansion value. Starting from the current region box, the program tries to expand the total width and total height by the given amount, while staying inside the image boundary.
  If this value is less than or equal to 0, the crop box is not expanded.
  Since the region boxes produced by the model are usually fairly tight around the subject, setting this above 0 can help preserve a reasonable amount of margin.

- `overflow_mode`: What to do when the crop box goes beyond the image boundary:

  - `"crop"`: Crop inward. No white border, but composition may be lost.
  - `"pad"`: Add white padding, preserving composition and aspect ratio.
  - `"auto"`: Switch automatically based on the configured loss threshold.

- `aspect_ratio_fix`:

  - `["0"]`: Keep the original box ratio.
  - `["2:3"]`: Force a fixed ratio.
  - `["512,768"]`: Force a target size and try to output that exact pixel size.
  - Multiple specs may be listed at the same time, and the program will export one crop result for each valid spec. For example:
    `["0", "2:3", "3:4", "768,768", "1024,1024"]`

## 7. Annotation Visuals

These settings affect only the preview style when `box_mode = true`.

- `main_box_color`: Main box color, using RGBA.

- `line_width`: Main box line width.

- `line_box_width`: Line width for the region boxes.

- `label_bg_color`: Label background color.

- `label_font_size`: Label font size.

- `label_padding_x`: Left and right padding inside labels.

- `label_padding_y`: Top and bottom padding inside labels.

- `label_gap`: Distance between the label and the anchor box.

- `font_candidates`: Ordered list of fonts the program tries to load.

- `[annotation.region_colors]`: Lets you assign a dedicated preview color to each region.

## Common Configuration Templates

### Template A: Quick Detection Check

```toml
box_mode = true
crop_mode = false

[save]
output_format = "jpg"
jpeg_quality = 85

[box_settings.full]
enabled = true
expand = 40
overflow_mode = "auto"
aspect_ratio_fix = ["0"]
```

Effect:

- Outputs preview images only.
- Uses smaller JPG files for quick inspection.
- Does not output crops.

### Template B: Fixed-Size Dataset Production (LoRA Training)

```toml
box_mode = false
crop_mode = true

[save]
output_format = "png"

[crop_logic]
size_expand_threshold_percent = 25
upscale_small_outputs = true

[box_settings.full]
enabled = true
expand = 60
overflow_mode = "pad"
aspect_ratio_fix = ["512,768", "512,512"]
```

Effect:

- Outputs crops only.
- Tries to produce fixed-size results.
- Uses white padding when needed to preserve composition.
- Allows final upscaling if the box still cannot reach the target size.

### Template C: Conservative Crop (No White Borders)

```toml
[crop_logic]
size_expand_threshold_percent = 0
upscale_small_outputs = false

[aspect_ratio_overflow]
auto_crop_max_loss_percent = 5

[box_settings.waist]
enabled = true
expand = 12
overflow_mode = "crop"
aspect_ratio_fix = ["2:3"]
```

Effect:

- Tries to avoid white borders.
- Keeps composition more conservative and closer to the source image.
- Does not aggressively enlarge boxes or upscale just to hit a fixed target size.
