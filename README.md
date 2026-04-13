# SoloCropper

SoloCropper is an AI-driven image processing utility designed for precise human-centric cropping. Built on Ultralytics pose and segmentation frameworks, it supports high-volume batch processing and generates multiple aspect ratios and sizes in a single pass.

It supports various input and output formats and is currently available as a command-line tool only.

For more LoRAs and articles, visit [**sololo.xyz**](https://sololo.xyz)



## Requirements

- Python `3.11+`
- Windows, Linux, or macOS
- PyTorch

## Model Downloads

Download the following model files and place them in `models/yolo/`:

- `yolo26x-seg.pt`
- `yolo26x-pose.pt`

GitHub:

- `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt`

- `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt`

Hugging Face:

- `https://huggingface.co/openvision/yolo26-x-seg`

- `https://huggingface.co/openvision/yolo26-x-pose`

Other YOLO segmentation and pose models are also supported. Update `config.toml` if you use different files.

## Installation

- Clone this repository.

  ```
  git clone https://github.com/sololo-xyz/SoloCropper.git
  ```

- Navigate to the project directory.

  ```
  cd SoloCropper
  ```

- Create and activate a Python virtual environment.

  #### Windows

  ```
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```

  #### Linux / macOS

  ```
  python3 -m venv venv
  . ./venv/bin/activate
  ```

  

- Install either the CPU or GPU version of PyTorch.

  

  CPU:

  ```
  pip install torch torchvision
  ```

  GPU:
  Follow the official PyTorch installation guide: `https://pytorch.org/get-started/locally/`

  

- Install the project dependencies.

  ```
  pip install -r requirements.txt
  ```

  

## Usage

### Windows

```powershell
.\run.bat
```

### Linux / macOS

```bash
chmod +x run.sh
./run.sh
```



## Configuration

Edit `config.toml` to change device, models, paths, thresholds, box settings, and output behavior.

- Configuration guide: [`docs/Config-Guide-EN.md`](docs/Config-Guide-EN.md)
