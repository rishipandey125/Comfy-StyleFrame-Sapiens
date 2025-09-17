PREPROCESS_MEAN = (123.5, 116.5, 103.5)
PREPROCESS_STD = (58.5, 57.0, 57.5)
PREPROCESS_H = 1024
PREPROCESS_W = 768


# downloads
DETECTORS = {
    "mmdet": {
        "repo": "facebook/sapiens-pose-bbox-detector",
        "name": "detector",
        "filename": "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
        "use_torchscript": False,
    },
    "yolo": {
        "repo": "Ultralytics/YOLOv8",
        "name": "detector",
        "filename": "yolov8m.pt",
        "use_torchscript": False,
    },
}
