# 🚀 YOLOv8-TritonServer-Python-Post-Processing-Pipeline

This repository hosts a robust and high-performance solution for deploying **YOLOv8** object detection models using **NVIDIA Triton Inference Server**. The pipeline leverages the **Ensemble Model** feature to tightly integrate the raw model inference (via ONNX/PyTorch backend) with a custom, highly optimized **Python Backend** for all post-processing steps (NMS, coordinate transforms, and output formatting).

The key benefit of this architecture is **reduced I/O overhead** by performing NMS directly on the server, rather than transferring large raw tensor outputs back to the client.

---

## ✨ Core Features & Technology Stack

- **Model:** [YOLOv8](https://docs.ultralytics.com/models/yolov8/) (easily adaptable to YOLOv5, YOLOv7, etc.)
- **Server:** [NVIDIA Triton Inference Server](https://github.com/triton-inference-server)
- **Inference Backend:** ONNX Runtime or PyTorch Backend
- **Post-processing:** Triton Python Backend (implements custom NMS logic tailored for modern YOLO outputs)
- **Pipeline:** Triton Ensemble Model for seamless execution flow.

---

## 📂 Repository Structure

```bash
model_repository/
├── README.md # This file
├── grpc_client.py # gRPC client script for inference and visualization
├── http_client.py # HTTP client script for inference and visualization
├── yolov8n_ensemble  # Configuration that chains the entire pipeline
│   ├── 1
│   └── config.pbtxt
├── yolov8n_onnx   # Model configuration
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
└── yolov8n_postprocess # Python backend for NMS and formatting
    ├── 1
    │   └── model.py # The core post-processing implementation
    └── config.pbtxt
```

---

## 🛠️ Setup and Deployment

### 1. Model Preparation

Place your trained YOLO model (e.g., `yolov8n.onnx` or `yolov8n.pt`) into the appropriate version subdirectory within `model_repository` (e.g., `model_repository/yolo11n_onnx/1/`).

### 2. Post-processing Logic (`model.py`)

The `model_repository/yolo_postprocess/1/model.py` file contains the essential NMS logic. Crucially, it includes adaptations for modern YOLO models:

1.  **Shape Correction:** Transforms the raw model output (e.g., `[1, C, N]`) into the NMS-ready `[N, C]` shape.
    ```python
    # In model.py's execute function
    predictions = np.squeeze(raw_output, axis=0).T
    ```
2.  **YOLOv8 NMS:** The logic correctly handles models where class scores immediately follow coordinates, eliminating the incorrect reliance on a separate objectness score.

### 3. Launching Triton Server

Start the Triton Server using Docker, ensuring the model repository is mounted and performance flags are utilized for optimal speed:

```bash
# Recommended command for performance and access
docker run --gpus all -it --rm \
    -p 8000:8000 \ # http port
    -p 8001:8001 \ # grpc port
    -p 8002:8002 \ # metric port
    --shm-size=1g --ulimit memlock=-1 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:<VERSION>-py3 \
    tritonserver --model-repository=/models
```

## 💡 Usage

### 1. Run Client

The client.py script performs Letterbox pre-processing, sends the image to the Triton Ensemble model via HTTP, and visualizes the results using human-readable class names (configured using the COCO 80 class list in the client).

Install dependencies:

```bash
pip install tritonclient[http] numpy opencv-python
```

Execute the client:

```bash
python client.py
```

### 2. Output Format

The client receives the final, formatted bounding boxes from the Ensemble model (yolo11n_ensemble). Each row contains the following 7 elements:

| Index | Name     | Description                                       |
| ----- | -------- | ------------------------------------------------- |
| 0     | image_id | Batch ID (0 for single-image batch)               |
| 1     | label_id | Detected class ID                                 |
| 2     | conf     | Final confidence score                            |
| 3     | xmin     | Top-left X coordinate (on the original image)     |
| 4     | ymin     | Top-left Y coordinate (on the original image)     |
| 5     | xmax     | Bottom-right X coordinate (on the original image) |
| 6     | ymax     | Bottom-right Y coordinate (on the original image) |

The final result will be saved to output_detection.jpg.

## 📌 Troubleshooting

- IndexError or Incorrect Output: This is typically a shape or logic error in model.py. Always fully restart the Triton Server after modifying the Python Backend code to ensure the changes are loaded.

- Connection Refused: Verify the IP address and port (e.g., 8000/8001) used in client.py match the port exposed by your Triton Server container.

- W1001 pinned_memory... Warning: This is a performance warning, not a functional error. It can be resolved by adding --shm-size=1g --ulimit memlock=-1 to your docker run command.
