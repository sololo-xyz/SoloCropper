# SoloCropper

**SoloCropper** is a high-efficiency **human** image **cropping tool** tailored for **dataset preparation**. It automates the generation of multi-spec outputs from multiple images, allowing users to crop various body segments and sizes in a single pass based on custom parameters.

## 🚀 Key Features
- Fast & Accurate Detection: Powered by high-efficiency Ultralytics YOLO models, SoloCropper delivers rapid human recognition even on standard CPUs.
- Flexible multi-output cropping: Generate multiple candidate crops at once to streamline your dataset selection process.
- Flexible Image Handling: Comprehensive support for multiple input/output formats and adjustable compression quality for precise output control.

## 🛠️ Project Status
SoloCropper is part of the Solo series by sololo.xyz. Currently available as a CLI-only version.

**Explore more and download hundreds of free Solo LoRAs at: https://sololo.xyz**



## 📋 Requirements

- Python `3.11+`
- Windows, Linux, or macOS
- PyTorch

## 📥 Model Downloads

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

## ⚙️ Installation

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

  

## ▶️ Usage

### Windows

```powershell
.\run.bat
```

### Linux / macOS

```bash
chmod +x run.sh
./run.sh
```



## 🔧 Configuration

Edit `config.toml` to change device, models, paths, thresholds, box settings, and output behavior.

- Configuration guide: [`docs/Config-Guide-EN.md`](docs/Config-Guide-EN.md)
