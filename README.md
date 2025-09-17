# Sapiens pose for ComfyUI

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)


<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/565c9438-1c2c-445d-89c0-df15ac3d8829" alt="Sapiens Pose Demo">
  <em>Demo</em>
</p>
<br>


This set of nodes integrates the **pose estimation** capabilities of [Sapiens](https://www.meta.com/emerging-tech/codec-avatars/sapiens/?utm_source=github.com&utm_medium=redirect). Currently, it focuses on accurately detecting and drawing human poses from input images or videos, providing robust and performant keypoint generation.

## ⚙️ Installation

1.  **Clone the Repository**
    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git ComfyUI_SapiensPose
    ```

2.  **Install Dependencies**
    Navigate into the cloned directory and install the required Python packages:
    ```bash
    cd ComfyUI_SapiensPose
    pip install -r requirements.txt
    ```

    ####  for Windows Users
    The `mmcv` and `mmdet` libraries, which are critical for the person detection pipeline, can be notoriously difficult to install on Windows. To simplify this, the `requirements.txt` file points to pre-built wheels for common Python/CUDA versions hosted in this repository. If you encounter issues, please check the `wheels` directory or open an issue. 

## 🚀 Usage

You can use this project in two ways: through ComfyUI or the command-line interface.

### 1. ComfyUI Workflow

The nodes are designed to be chained together in a logical pipeline:

1.  **Load Sapiens Model**: Loads the pose estimation and person detection models into memory. It will handle downloading them for you.
2.  **Sapiens Pose Pre-Process**: Takes a batch of images and detects bounding boxes for each person.
3.  **Sapiens Pose Estimate**: Runs the core Sapiens model on the pre-processed data to generate keypoints.
4.  **Sapiens Draw Pose**: Draws the final skeletons onto the original images or a black background.

#### Manual Model Placement

If you prefer to manage models manually and avoid auto-downloading, place them in the following structure inside your `ComfyUI/models/` directory:

```
📂 ComfyUI/  
├── 📂 models/  
│   ├── 📂 sapiens/  
│   │   └── 📂 pose/  
│   │       └── sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2
│   │   └── 📂 detector/  
│   │       └── rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
```



### 2. Command-Line Interface (CLI)

The CLI is perfect for batch processing, automation, or for users who don't use ComfyUI.

#### CLI Installation
Install the extra dependencies required for the CLI:
```bash
pip install typer rich imageio imageio[pyav] huggingface-hub tqdm requests
```

#### CLI Examples

> [!NOTE]  
> The first time you run a command, the necessary models will be downloaded and cached to `~/.cache/sapiens-pose-cli/` by default.

**Basic Usage:**
Process a video and save the result.
```bash
python cli.py path/to/your/video.mp4 --output result.mp4
```

**Get Help:**
See all available options and commands.
```bash
python cli.py --help
```

**Advanced Usage:**
Process the first 150 frames using the faster extraction mode and thicker bones.
```bash
python cli.py "path/to/my dance video.mov" \
    --output "dance_skeletons.mp4" \
    --extract-mode "fast" \
    --bone-thickness 5 \
    --frame-limit 150
```

# Render on a black background
```bash
python cli.py path/to/your/video.mp4 \
    --output "result_black_bg.mp4" \
    --load-keypoints "my_keypoints.pt" \
    --draw-on-black
```


## 📦 Models

This project relies on two primary models:

-   **Pose Estimation**: `sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2`
    -   A high-performance version of the Sapiens model, exported to `bfloat16` using a modern `torch.export` pipeline for maximum speed on compatible hardware. The official Meta-provided `bfloat16` model uses an older TorchScript format.
    -   *Source*: [melmass/sapiens on Hugging Face](https://huggingface.co/melmass/sapiens/blob/main/sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2)
    - if your hardware doesn't support `bfloat16` you can use the [torchscript version](https://huggingface.co/facebook/sapiens-pose-1b-torchscript)

-   **BBox Detection**: `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth`
    -   An RTMDet model fine-tuned for robust person detection.
    -   *Source*: [facebook/sapiens-pose-bbox-detector on Hugging Face](https://huggingface.co/facebook/sapiens-pose-bbox-detector/blob/main/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth)

## 📄 License

This project has a composite license structure. The code I have written is licensed under the MIT License, but the Sapiens models from Meta are distributed under a more restrictive license.

-   **This Repository's Code**: Licensed under the [MIT License](LICENSE).
-   **Sapiens Models & Weights**: Licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## Acknowledgements
- This work is built upon the incredible research and models released by Meta AI for **Sapiens**.
- The person detector uses the **mmdetection** framework.

