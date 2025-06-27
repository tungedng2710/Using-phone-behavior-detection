# Using Phone Behavior Detection

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tungn197/Using-phone-behavior-detection.git
    cd Using-phone-behavior-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Copy the `run.example.sh` to `run.sh`
From your Terminal, just run
```bash
bash run.sh
```

Example script:
```
python pipeline.py \
    --source "rtsp://admin:Admin123@10.255.186.102:554/Streaming/Channels/101" \
    --person_model_weights weights/yolov8s.pt \
    --phone_model_weights weights/yolo12l.pt \
    --show_streaming --conf_thresh 0.25
```

the `source` can be webcam (0), rtsp url or file path

The pipeline keeps the bounding box of any person detected using a phone for five seconds even if the phone is not detected in subsequent frames.
