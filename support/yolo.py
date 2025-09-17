from dataclasses import dataclass
import torch

from . import datatypes as dt


@dataclass
class DetectorConfig:
    model_path: str = "models/yolov8l.pt"
    person_id: int = 0
    conf_thres: float = 0.25


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    import cv2

    draw_img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        draw_img = cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)
    return draw_img


class Detector:
    def __init__(self, config: DetectorConfig = DetectorConfig()):
        from ultralytics import YOLO

        model_path = config.model_path
        if not model_path.endswith(".pt"):
            model_path = model_path.split(".")[0] + ".pt"
        self.model = YOLO(model_path)
        self.person_id = config.person_id
        self.conf_thres = config.conf_thres

    def __call__(self, img: torch.Tensor, padding: float | None = None) -> dt.BBox:
        return self.detect(img, padding)

    def _apply_padding(self, boxes: dt.BBox, padding: float):
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] -= w * padding
        boxes[:, 1] -= h * padding
        boxes[:, 2] += w * padding
        boxes[:, 3] += h * padding
        return boxes

    def detect(
        self,
        img: torch.Tensor,
        confidence: float | None = None,
        padding: float | None = None,
    ) -> dt.BBox:
        with torch.no_grad():
            results = self.model(
                img, conf=confidence or self.conf_thres, classes=[0], verbose=False
            )
            # (x1, y1, x2, y2, conf, cls)
            output = []
            img_height = img.shape[2]
            img_width = img.shape[3]
            for result in results:
                detections = result.boxes.data  # .cpu().numpy()

                # Filter out only person
                # (classes kwarg does the filtering already)
                # person_detections = detections[detections[:, -1] == self.person_id]
                # boxes = person_detections[:, :4]
                boxes = detections[:, :4]

                # no person found
                if boxes.shape[0] == 0:
                    output.append(
                        torch.empty((0, 4), dtype=torch.int, device=img.device)
                    )
                    continue
                    # return torch.empty((0, 4), dtype=torch.int, device=img.device)

                if padding is not None:
                    boxes = self._apply_padding(boxes, padding)

                # round to int
                boxes_int = torch.round(boxes).int()

                # guarantees x2 > x1 and y2 > y1 without using masks
                boxes_int[:, 2] = torch.max(boxes_int[:, 2], boxes_int[:, 0] + 1)
                boxes_int[:, 3] = torch.max(boxes_int[:, 3], boxes_int[:, 1] + 1)

                # clamp boundaries
                min_vals = torch.zeros(4, dtype=torch.int, device=boxes_int.device)
                max_vals = torch.tensor(
                    [img_width - 1, img_height - 1, img_width - 1, img_height - 1],
                    dtype=torch.int,
                    device=boxes_int.device,
                )
                boxes_int = torch.max(min_vals, torch.min(boxes_int, max_vals))
                output.append(boxes_int)

        return output  # torch.cat(output, dim=0)

    def heatmaps_to_keypoints(
        self,
        heatmaps: torch.Tensor,
        bboxes: torch.Tensor,
        preprocessed_shape: tuple[int, int],
        original_shape: tuple[int, int],
        # *,
        # progress_callback: ProgressCallback | None = None,
        # chunks=32,
    ):
        bboxes = bboxes.to(heatmaps.device)

        # TODO: add chunks here
        with torch.no_grad():
            N, K, H_heatmap, W_heatmap = heatmaps.shape

            # flatten it
            # Shape: (N, K, H_h * W_h)
            flat_heatmaps = heatmaps.view(N, K, -1)

            # max_confidences shape: (N, K)
            # max_indices shape: (N, K)
            max_confidences, max_indices = torch.max(flat_heatmaps, dim=2)

            # 1D -> 2D coordinates
            x_on_heatmap = (max_indices % W_heatmap).float()
            y_on_heatmap = (max_indices // W_heatmap).float()

            x1, y1, x2, y2 = bboxes.T

            bbox_width = x2 - x1
            bbox_height = y2 - y1

            x_relative = x_on_heatmap / W_heatmap
            y_relative = y_on_heatmap / H_heatmap

            x_in_preprocessed_space = x1.unsqueeze(
                1
            ) + x_relative * bbox_width.unsqueeze(1)
            y_in_preprocessed_space = y1.unsqueeze(
                1
            ) + y_relative * bbox_height.unsqueeze(1)

            # scale
            preprocessed_h, preprocessed_w = preprocessed_shape
            original_h, original_w = original_shape

            scale_x = float(preprocessed_w) / original_w
            scale_y = float(preprocessed_h) / original_h

            # un-scale
            x_final = x_in_preprocessed_space / scale_x
            y_final = y_in_preprocessed_space / scale_y

            keypoints = torch.stack([x_final, y_final, max_confidences], dim=2)

            return keypoints
