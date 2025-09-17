# ruff: noqa: F722 - Syntax error in forward annotation: {parse_error}
from contextlib import suppress
import gc

import cv2
import psutil
import functools
import logging
import os
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms.functional as TF

from .proxy_tensor import is_tensor, tensor_info

from .datatypes import BBox, Float, Float32, UInt8

from typing import Generator, Literal, Protocol, Type, runtime_checkable, Any

import torch.nn.functional as F
from pathlib import Path
from typing import TypeVar, TypedDict
from enum import Enum

from .datatypes import KeypointResult, LinkInfo


from .classes_and_palettes import (
    GOLIATH_KEYPOINTS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    SKELETON_LINKS_BY_INDEX,
    GOLIATH_MEMBER_GROUPS_BY_INDEX,
)

from torchvision import transforms

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def to_np(tensor: Any):
    if not is_tensor(tensor):
        raise ValueError(f"to_np only supports tensors, you passed {type(tensor)}")

    dtype_mult = 1 if tensor.dtype == torch.uint8 else 255
    return (tensor.cpu().numpy() * dtype_mult).astype(np.uint8)


def midpoint(A: tuple[int, int], B: tuple[int, int]):
    return ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)


def setup_models():
    import folder_paths

    weigths_path = Path(folder_paths.models_dir) / "sapiens"
    folder_paths.add_model_folder_path(
        "sapiens", weigths_path.as_posix(), is_default=False
    )
    weigths_path.mkdir(exist_ok=True)

    # for n in ["seg", "depth", "pose", "normal"]:
    #    (weigths_path / n).mkdir(exist_ok=True)


# common types
class ModelMeta(TypedDict):
    name: str
    repo: str
    filename: str
    use_torchscript: bool


class TaskType(Enum):
    DEPTH = "depth"
    NORMAL = "normal"
    SEG = "seg"
    POSE = "pose"


@runtime_checkable
class ProgressCallback(Protocol):
    def __call__(self, step: int, **kwargs: Any) -> None:
        """
        A protocol for progress callback functions.

        Args:
            step (int): The progress value or step count.
            **kwargs: Can include 'title' (str) and 'subtitle' (str).
        """
        pass


def pad_tensor_to_batch_size(
    tensor: torch.Tensor, target_batch_size: int
) -> torch.Tensor:
    """
    Pads a tensor along its first (batch) dimension to a target size.
    """
    current_batch_size = tensor.shape[0]
    padding_needed = target_batch_size - current_batch_size

    if padding_needed <= 0:
        return tensor

    # Create a padding tensor with the same shape as the original,
    # but with the batch dimension changed to the amount of padding needed.
    padding_shape = (padding_needed,) + tensor.shape[1:]
    padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)

    return torch.cat([tensor, padding], dim=0)


def chunked_tensor(tensor: torch.Tensor, chunk_size=16):
    B_full = tensor.shape[0]
    n_chunks = (B_full + chunk_size - 1) // chunk_size

    for _i in range(n_chunks):
        start_idx = _i * chunk_size
        end_idx = min((_i + 1) * chunk_size, B_full)
        _slice = tensor[start_idx:end_idx]
        valid_len = _slice.shape[0]
        _padded = pad_tensor_to_batch_size(_slice, chunk_size)
        yield _padded, valid_len


T_Chunk = TypeVar("T_Chunk")


# @profile
def chunked_tensors(
    chunk_class: Type[T_Chunk],
    *,
    chunk_size: int = 16,
    auto_release=True,
    **tensors: torch.Tensor | list[torch.Tensor],
) -> Generator[tuple[T_Chunk, int], None, None]:
    """
    Chunks multiple named tensors along their first dimension and yields them
    as a namedtuple.

    This generator takes keyword arguments for tensors, validates they have the
    same batch size, and then yields chunks of a specified size. The last chunk
    is padded with zeros to match the chunk_size.

    Args:
        chunk_size (int): The desired size for the first dimension of each chunk.
        **tensors (torch.Tensor): A variable number of keyword arguments where each
            value is a tensor to be chunked. All tensors must have the same
            size in their first dimension.

    Yields:
        tuple[namedtuple, int]: A tuple containing:
        - A namedtuple where each field corresponds to a keyword argument and
          holds the padded tensor chunk (e.g., `chunk.image`, `chunk.mask`).
        - An integer representing the number of valid (non-padded) items
          in the chunk.
    """
    if not tensors:
        # If no tensors are provided, the generator simply finishes.
        return

    tensor_names = sorted(list(tensors.keys()))

    actual_fields = sorted(
        list(set(getattr(chunk_class, "__annotations__", {}).keys()))
    )
    if tensor_names != actual_fields:
        raise TypeError(
            f"Field mismatch: chunk_class '{chunk_class.__name__}' has fields {actual_fields}, "
            f"but received tensor arguments {tensor_names}."
        )

    tensor_values = list(tensors.values())
    # --- Validation ---
    if isinstance(tensor_values[0], torch.Tensor):
        B_full = tensor_values[0].shape[0]
    else:
        # B_full = max([x.shape[0] for x in tensor_values[0]])
        B_full = len(tensor_values[0])

    for name, t in tensors.items():
        if isinstance(t, torch.Tensor):
            if t.shape[0] != B_full:
                raise ValueError(
                    f"All tensors must have the same size in the first dimension. "
                    f"Tensor '{tensor_names[0]}' has size {B_full}, but "
                    f"tensor '{name}' has size {t.shape[0]}."
                )
        else:
            # B_cur = max([x.shape[0] for x in t])
            B_cur = len(t)
            if B_cur != B_full:
                raise ValueError(
                    f"All tensors must have the same size in the first dimension. "
                    f"Tensor '{tensor_names[0]}' has size {B_full}, but "
                    f"tensor '{name}' has size {B_cur}."
                )

    n_chunks = (B_full + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, B_full)

        # Create a dictionary of the sliced tensors
        slices = {
            name: pad_tensor_to_batch_size(t[start_idx:end_idx], chunk_size)
            for name, t in tensors.items()
            if isinstance(t, torch.Tensor)
        } | {
            name: pad_tensor_to_batch_size(
                pad_sequence(t[start_idx:end_idx], batch_first=True, padding_value=0.0),
                chunk_size,
            )
            for name, t in tensors.items()
            if not isinstance(t, torch.Tensor)
        }
        valid_len = next(iter(slices.values())).shape[0]

        # Instantiate the user's class with the padded chunk data
        chunk_instance = chunk_class(**slices)
        yield chunk_instance, valid_len

        if auto_release:
            for s in slices.values():
                del s
            del chunk_instance, valid_len
            torch.cuda.empty_cache()
            gc.collect()


def print_ram(step=""):
    print(
        f"{step}\nRAM usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB"
    )


def _apply_padding_and_clamp(
    bboxes: Float[torch.Tensor, "N 4"],
    padding_ratio: float,
    image_shape: tuple[int, int] | torch.Size,
) -> torch.Tensor:
    """
    Expands a batch of bounding boxes by a given ratio and clamps them to
    the image boundaries.

    Args:
        bboxes (torch.Tensor): A tensor of shape (N, 4) with (x1, y1, x2, y2).
        padding_ratio (float): The ratio to expand the boxes by (e.g., 0.1 for 10%).
        image_shape (tuple[int, int]): The (Height, Width) of the image for clamping.

    Returns:
        torch.Tensor: The padded and clamped bounding boxes, shape (N, 4).
    """
    img_h, img_w = image_shape

    x1, y1, x2, y2 = bboxes.T

    # calculate width, height, and the padding amount for each dimension
    width = x2 - x1
    height = y2 - y1
    pad_w = width * padding_ratio
    pad_h = height * padding_ratio

    new_x1 = x1 - pad_w
    new_y1 = y1 - pad_h
    new_x2 = x2 + pad_w
    new_y2 = y2 + pad_h

    padded_bboxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

    min_vals = torch.zeros(4, device=bboxes.device, dtype=bboxes.dtype)
    max_vals = torch.tensor(
        [img_w, img_h, img_w, img_h], device=bboxes.device, dtype=bboxes.dtype
    )

    clamped_bboxes = torch.max(min_vals, torch.min(padded_bboxes, max_vals))

    return clamped_bboxes


# @jaxtyped(typechecker=typechecker)
def detect_and_crop_people(
    input_tensor: Float[torch.Tensor, "B H W C"],
    chunk_size: int,
    *,
    input_bboxes: Float[np.ndarray, "_B N 4"] | None = None,
    confidence: float = 0.25,
    padding: float = 0.0,
    progress_callback=None,
    skip_preprocessor=False,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Detects people in a large tensor, crops them, and returns both the
    cropped image data and the corresponding bounding box metadata.

    Args:
        input_tensor: A tensor of images (B, C, H, W).
        chunk_size: The size of chunks to process at a time.

    Returns:
        A tuple containing:
        - all_bboxes (torch.Tensor): Shape (N, 5), format [batch_idx, x1, y1, x2, y2].
                                     The batch_idx is absolute to the input_tensor.
        - all_crops (torch.Tensor): Shape (N, C, H_out, W_out) of cropped people.
    """
    import torchvision

    device = (
        device or torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    all_final_bboxes = []
    all_final_crops = []

    # if input_bboxes is None:
    from .yolo import Detector

    # absolute frame index offset
    processed_frames_offset = 0

    preprocessor = pose_preprocessor()
    if input_bboxes is None:
        detector = Detector()

    for chunk in torch.split(input_tensor, chunk_size, dim=0):
        chunk = chunk.to(device)
        if not skip_preprocessor:
            preprocessed_chunk = preprocessor(chunk.permute(0, 3, 1, 2))
        else:
            preprocessed_chunk = ensure_stride(chunk).permute(0, 3, 1, 2)

        if input_bboxes is None:
            list_of_boxes_for_chunk = detector.detect(
                preprocessed_chunk, confidence=confidence
            )
        else:
            list_of_boxes_for_chunk = input_bboxes[: chunk.shape[0]]

        boxes_for_roi = []
        for i, boxes_for_one_image in enumerate(list_of_boxes_for_chunk):
            if boxes_for_one_image.shape[0] > 0:
                num_boxes = boxes_for_one_image.shape[0]
                local_batch_indices = torch.full(
                    (num_boxes, 1), fill_value=i, device=device
                )
                boxes_with_indices = torch.cat(
                    [
                        local_batch_indices.float(),
                        boxes_for_one_image.to(device).float(),
                    ],
                    dim=1,
                )
                boxes_for_roi.append(boxes_with_indices)

        if not boxes_for_roi:
            processed_frames_offset += chunk.shape[0]
            continue

        tight_boxes_local = torch.cat(boxes_for_roi, dim=0)

        if padding > 0.0:
            # Get the shape of the image the boxes were detected on
            image_shape_for_padding = preprocessed_chunk.shape[2:]  # (H, W)

            # Apply padding and clamping to the coordinate part of the boxes
            padded_coords = _apply_padding_and_clamp(
                tight_boxes_local[:, 1:],  # only (x1,y1,x2,y2)
                padding,
                image_shape_for_padding,
            )
            # re-attach the local frame index for roi_align
            boxes_for_cropping = torch.cat(
                [tight_boxes_local[:, 0:1], padded_coords], dim=1
            )
        else:
            boxes_for_cropping = tight_boxes_local

        cropped_people = torchvision.ops.roi_align(
            input=preprocessed_chunk,
            boxes=boxes_for_cropping,
            output_size=(1024, 768),
            spatial_scale=1.0,
            aligned=True,
        )
        all_final_crops.append(cropped_people)

        del cropped_people

        padded_boxes_absolute = boxes_for_cropping.clone()
        padded_boxes_absolute[:, 0] += processed_frames_offset
        all_final_bboxes.append(padded_boxes_absolute)

        del padded_boxes_absolute

        processed_frames_offset += chunk.shape[0]

        if progress_callback:
            progress_callback(1)

    if not all_final_crops:
        C, H_out, W_out = input_tensor.shape[1], 1024, 768
        return torch.empty((0, 5), device=input_tensor.device), torch.empty(
            (0, C, H_out, W_out), device=input_tensor.device
        )

    return torch.cat(all_final_bboxes, dim=0).to("cpu"), torch.cat(
        all_final_crops, dim=0
    ).to("cpu")


def ensure_stride(tensor: torch.Tensor, stride: int = 32) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(
            f"Input tensor must be BCHW (4 dimensions), but got {tensor.dim()} dimensions."
        )

    _B, H, W, _C = tensor.shape

    pad_h = 0
    if H % stride != 0:
        pad_h = stride - (H % stride)

    pad_w = 0
    if W % stride != 0:
        pad_w = stride - (W % stride)

    if pad_h == 0 and pad_w == 0:
        log.info(
            f"Image dimensions {H}x{W} are already divisible by {stride}. No padding needed."
        )
        return tensor

    log.warning(
        f"Image dimensions {H}x{W} are not divisible by {stride}. "
        f"Padding to {H + pad_h}x{W + pad_w} (H_pad={pad_h}, W_pad={pad_w})."
    )

    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left

    padded_img = F.pad(
        tensor,
        (0, 0, pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
        mode="constant",
        value=0,
    )

    log.info(f"Padded image shape: {padded_img.shape}")
    return padded_img


def pose_preprocessor(
    input_size: tuple[int, int] = (768, 1024),
    # mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    # std: tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    return transforms.Compose(
        [
            transforms.Resize(input_size),
            # transforms.Normalize(mean=mean, std=std),
        ]
    )


def calculate_proportional_radius(
    image_shape: tuple[int, int], base_radius: int = 3, *, base=768
) -> int:
    """
    Calculate proportional radius based on image dimensions.
    Uses the smaller dimension (width or height) as reference.
    """
    H, W = image_shape
    min_dim = min(H, W)
    scale_factor = min_dim / base
    return max(1, int(base_radius * scale_factor))


def _group_data_by_frame(
    num_frames: int, abs_keypoints: np.ndarray, bboxes_with_frame_idx: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Groups keypoints and bounding boxes by the frame they belong to.

    Returns:
        frame_to_kpts: A list where list[i] is an array of keypoints (P, K, 3) for people in frame i.
        frame_to_bboxes: A list where list[i] is an array of bboxes (P, 4) for people in frame i.
    """
    frame_to_kpts_list = [[] for _ in range(num_frames)]
    frame_to_bboxes_list = [[] for _ in range(num_frames)]

    person_to_frame_map = bboxes_with_frame_idx[:, 0]
    bbox_coords = bboxes_with_frame_idx[:, 1:]

    for person_idx, frame_idx in enumerate(person_to_frame_map):
        frame_idx = int(frame_idx)
        if frame_idx < num_frames:
            frame_to_kpts_list[frame_idx].append(abs_keypoints[person_idx])
            frame_to_bboxes_list[frame_idx].append(bbox_coords[person_idx])

    frame_to_kpts = [
        np.stack(kpts) if kpts else np.empty((0, abs_keypoints.shape[1], 3))
        for kpts in frame_to_kpts_list
    ]
    frame_to_bboxes = [
        np.stack(bboxes) if bboxes else np.empty((0, 4))
        for bboxes in frame_to_bboxes_list
    ]

    return frame_to_kpts, frame_to_bboxes


# OLD
# @jaxtyped(typechecker=typechecker)
def draw_keypoints_chunked(
    imgs: np.ndarray,  # (B, H, W, C), float32, [0,1], RGB
    abs_keypoints: np.ndarray,  # (N_people, K_kpts, 3) with (x, y, conf)
    bboxes_with_frame_idx: np.ndarray,  # (N_people,) with [frame_idx, x1, y1, x2, y2]
    chunk_size: int = 16,
    kpt_threshold: float = 0.3,
    kpt_radius: int = 3,
    line_thickness: int = 5,
    draw_bbox: bool = False,
) -> np.ndarray:
    """
    Draws keypoints and skeletons on a batch of images, processing in chunks.
    """
    import cv2

    B, H, W, C = imgs.shape

    frame_to_kpts, frame_to_bboxes = _group_data_by_frame(
        B, abs_keypoints, bboxes_with_frame_idx
    )

    imgs_out = np.empty_like(imgs)
    auto_radius = calculate_proportional_radius((H, W), kpt_radius)
    auto_thickness = calculate_proportional_radius((H, W), line_thickness)

    for i in range(0, B, chunk_size):
        chunk_slice = slice(i, i + chunk_size)
        img_chunk = (imgs[chunk_slice].copy() * 255).astype(np.uint8)

        for j in range(img_chunk.shape[0]):
            abs_frame_idx = i + j
            img_to_draw_on = img_chunk[j]

            # get data for this frame
            keypoints_in_frame = frame_to_kpts[abs_frame_idx]
            bboxes_in_frame = frame_to_bboxes[abs_frame_idx]

            # loop over bbox of frame
            for person_idx in range(keypoints_in_frame.shape[0]):
                keypoints = keypoints_in_frame[person_idx]  # Shape (K, 3)

                if draw_bbox:
                    bbox = bboxes_in_frame[person_idx]
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(
                        img_to_draw_on,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        auto_thickness // 2,
                    )

                for link_info in SKELETON_LINKS_BY_INDEX:
                    pt1_idx, pt2_idx = link_info["link_indices"]
                    if (
                        keypoints[pt1_idx, 2] > kpt_threshold
                        and keypoints[pt2_idx, 2] > kpt_threshold
                    ):
                        x1, y1, _ = keypoints[pt1_idx]
                        x2, y2, _ = keypoints[pt2_idx]
                        color_rgb = tuple(int(c) for c in link_info["color"])  # [::-1])
                        cv2.line(
                            img_to_draw_on,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color_rgb,
                            auto_thickness,
                        )

                for kpt_idx in range(keypoints.shape[0]):
                    if keypoints[kpt_idx, 2] > kpt_threshold:
                        x, y, _ = keypoints[kpt_idx]
                        color_rgb = tuple(
                            int(c)
                            for c in GOLIATH_KPTS_COLORS[
                                kpt_idx % len(GOLIATH_KPTS_COLORS)
                            ]
                        )
                        cv2.circle(
                            img_to_draw_on, (int(x), int(y)), auto_radius, color_rgb, -1
                        )

        imgs_out[chunk_slice] = img_chunk.astype(np.float32) / 255.0

    return imgs_out


# @jaxtyped(typechecker=typechecker)
def extract_bbox_sequences(
    frames: Float32[torch.Tensor, "B H W C"],
    bboxes_list: list[Float32[np.ndarray, "_N 4"]],
    *,
    # Desired (height, width) for all crops
    target_crop_size: tuple[int, int] = (1024, 768),
    # RGB color for blank frames, 0-255 range
    blank_color: tuple[int, int, int] = (0, 0, 0),
) -> list[torch.Tensor]:
    """
    Extracts sequences of cropped bounding boxes from video frames, maintaining
    their original index across frames (very approximate mapping unless using deep sort and the like)

    """
    _B, H, W, C = frames.shape
    device = frames.device
    dtype = frames.dtype

    # the maximum number of unique bounding boxes across all frames.
    max_bboxes_per_frame = 0
    for frame_bboxes in bboxes_list:
        if frame_bboxes.shape[0] > max_bboxes_per_frame:
            max_bboxes_per_frame = frame_bboxes.shape[0]

    if max_bboxes_per_frame == 0:
        log.warning(
            "Warning: No bounding boxes found in any frame. Returning empty list."
        )
        return []

    output_sequences: list[list[torch.Tensor]] = [
        [] for _ in range(max_bboxes_per_frame)
    ]

    if dtype == torch.uint8:
        fill_value_tensor = torch.tensor(
            blank_color, dtype=dtype, device=device
        ).reshape(1, 1, C)
        blank_frame_tensor = fill_value_tensor.expand(
            target_crop_size[0], target_crop_size[1], C
        ).clone()
    elif dtype == torch.float:
        fill_value_tensor = torch.tensor(
            [c / 255.0 for c in blank_color], dtype=dtype, device=device
        ).reshape(1, 1, C)
        blank_frame_tensor = fill_value_tensor.expand(
            target_crop_size[0], target_crop_size[1], C
        ).clone()
    else:
        blank_frame_tensor = torch.zeros(
            (target_crop_size[0], target_crop_size[1], C), dtype=dtype, device=device
        )

    for frame_tensor, current_frame_bboxes in zip(frames, bboxes_list):
        for bbox_id in range(max_bboxes_per_frame):
            if bbox_id < current_frame_bboxes.shape[0]:
                bbox = current_frame_bboxes[bbox_id]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)

                # clamp coordinates to frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)

                if x1 >= x2 or y1 >= y2:
                    cropped_tensor = blank_frame_tensor
                else:
                    cropped_tensor = frame_tensor[y1:y2, x1:x2, :]

                    # (C, H_crop, W_crop)
                    cropped_tensor = cropped_tensor.permute(2, 0, 1)

                    # (C, target_H, target_W)
                    cropped_tensor = TF.resize(cropped_tensor, list(target_crop_size))

                    # (target_H, target_W, C)
                    cropped_tensor = cropped_tensor.permute(1, 2, 0)
            else:
                # insert blank frame.
                cropped_tensor = blank_frame_tensor

            output_sequences[bbox_id].append(cropped_tensor)

    final_output: list[torch.Tensor] = [
        torch.stack(sequence_list, dim=0) for sequence_list in output_sequences
    ]

    return final_output


# @jaxtyped(typechecker=typechecker)
def draw_images(
    results: KeypointResult,
    *,
    img: Float[np.ndarray, "C H W"] | Float[torch.Tensor, "H W C"] | None = None,
    target_size: tuple[int, int] | None = None,
    draw_on_black: bool = True,
    verbose: bool = False,
    # heatmap_scale: int = 4,
    kpt_colors: list[tuple[int, int, int]] | None,
    kpt_thr: float = 0.3,
    radius: int = 4,
    skeleton_info: dict[int, LinkInfo] | None = None,
    thickness: int = 3,
    style: Literal["sapiens", "openpose"] = "sapiens",
    draw_face: Literal["off", "simple", "clean", "full"] = "full",
    draw_clavicle=False,
    draw_upper_body=True,
    draw_lower_body=True,
    draw_labels: Literal["off", "name", "id", "link"] | None = None,
    label_size=0.5,
    bitmask: torch.Tensor | None = None,
    draw_keypoints=True,
    draw_skeleton=True,
    draw_bbox=False,
    bboxes: BBox | None = None,
    bbox_color: tuple[int, int, int] = (0, 255, 0),
    bbox_thickness: int = 3,
    # input_shape: tuple[int, int] = (PREPROCESS_W, PREPROCESS_H),
) -> UInt8[np.ndarray, "H W C"] | None:
    """Draw all people on a single image.

    This method doesn't autoscale radius and thickness
    """

    kpt_colors = list(kpt_colors or GOLIATH_KPTS_COLORS)
    skeleton_info = dict(skeleton_info or GOLIATH_SKELETON_INFO)
    if verbose:
        print(tensor_info(img, "received"))

    ALLOWED_IDs = []
    SKIP_LINKS = []

    if draw_face in ["clean", "full"]:
        ALLOWED_IDs.extend(GOLIATH_MEMBER_GROUPS_BY_INDEX["face"])

    if draw_face == "simple":
        ALLOWED_IDs.extend([0, 1, 2, 3, 4])

    if draw_face == "clean":
        SKIP_LINKS.extend([12, 13, 14, 15, 16])

    if draw_clavicle:
        ALLOWED_IDs.extend(GOLIATH_MEMBER_GROUPS_BY_INDEX["clavicle"])
    else:
        SKIP_LINKS.extend([17, 18])

    if draw_upper_body:
        ALLOWED_IDs.extend(GOLIATH_MEMBER_GROUPS_BY_INDEX["upper_body"])

    if draw_lower_body:
        ALLOWED_IDs.extend(GOLIATH_MEMBER_GROUPS_BY_INDEX["lower_body"])

    if style == "openpose":
        # middle face bone
        SKIP_LINKS.extend([12])

        # body sides
        SKIP_LINKS.extend([4, 5, 6])

        # lashline
        for x in range(96, 104 + 1):
            with suppress(ValueError):
                ALLOWED_IDs.remove(x)

        for x in range(120, 128 + 1):
            with suppress(ValueError):
                ALLOWED_IDs.remove(x)

        for x in range(144, 152 + 1):
            with suppress(ValueError):
                ALLOWED_IDs.remove(x)

        for x in range(161, 177 + 1):
            with suppress(ValueError):
                ALLOWED_IDs.remove(x)
        for x in range(220, 280 + 1):
            with suppress(ValueError):
                ALLOWED_IDs.remove(x)


        ALLOWED_IDs.remove(63)
        ALLOWED_IDs.remove(64)

        ALLOWED_IDs.remove(65)
        ALLOWED_IDs.remove(66)
        ALLOWED_IDs.remove(69)

        ALLOWED_IDs.remove(76)
        ALLOWED_IDs.remove(77)
        # OVERRIDES
        kpt_colors[21:63] = [(0, 0, 255)] * (63 - 21)  # Blue

        kpt_colors[70:220] = [(255, 255, 255)] * (220 - 70)  # White
        kpt_colors[0] = (255, 0, 0)  # Red

        kpt_colors[3:5] = [(255, 255, 255)] * 2  # Red

        kpt_colors[1] = (241, 0, 241)  # Purple
        kpt_colors[2] = (173, 9, 255)  # Purple
        kpt_colors[3] = (255, 0, 85)  # Red
        kpt_colors[4] = (255, 0, 170)  # Pink
        kpt_colors[5] = (85, 255, 0)  # Green
        kpt_colors[6] = (255, 170, 0)  # Gold
        kpt_colors[7] = (0, 255, 0)  # Green
        kpt_colors[8] = (255, 255, 0)  # Yellow
        kpt_colors[10] = (0, 255, 170)  # Green

        skeleton_info[1]["color"] = (0, 102, 153)  # Dark teal
        skeleton_info[3]["color"] = (68, 147, 66)  # Green
        skeleton_info[8]["color"] = (128, 255, 0)

        skeleton_info[11]["color"] = (255, 255, 0)

        skeleton_info[0]["color"] = (0, 51, 153)  # Dark blue
        # RLEG
        skeleton_info[3]["color"] = (0, 153, 51)  # Dark green  # Dark purple

        # LHAND
        skeleton_info[13]["color"] = (153, 0, 153)  # Dark purple
        skeleton_info[14]["color"] = (51, 0, 153)  # Dark blue
        skeleton_info[15]["color"] = (153, 0, 102)  # Plum
        skeleton_info[16]["color"] = (102, 0, 153)  # Dark purple

        skeleton_info[29]["color"] = (204, 255, 0)  # Lime
        skeleton_info[30]["color"] = (128, 255, 0)  # Green
        skeleton_info[31]["color"] = (51, 255, 0)  # Green
        skeleton_info[32]["color"] = (0, 255, 25)  # Green

        skeleton_info[33]["color"] = (0, 255, 102)  # Green
        skeleton_info[34]["color"] = (0, 255, 179)  # Green
        skeleton_info[35]["color"] = (0, 255, 255)  # Aqua
        skeleton_info[36]["color"] = (0, 178, 255)  # Turquoise

        skeleton_info[37]["color"] = (0, 102, 255)  # Blue
        skeleton_info[38]["color"] = (0, 25, 255)  # Blue
        skeleton_info[39]["color"] = (51, 0, 255)  # Blue
        skeleton_info[40]["color"] = (128, 0, 255)  # Purple

        skeleton_info[41]["color"] = (204, 0, 255)
        skeleton_info[42]["color"] = (255, 0, 230)
        skeleton_info[43]["color"] = (227, 0, 136)  # Pink
        skeleton_info[44]["color"] = (255, 0, 77)  # Red

        # RHAND
        skeleton_info[25]["color"] = (255, 0, 0)  # Red
        skeleton_info[26]["color"] = (255, 77, 0)  # Red
        skeleton_info[27]["color"] = (255, 153, 0)  # Orange
        skeleton_info[28]["color"] = (255, 229, 0)  # Yellow

        skeleton_info[45]["color"] = (255, 0, 0)  # Red
        skeleton_info[46]["color"] = (255, 77, 0)  # Red
        skeleton_info[47]["color"] = (255, 153, 0)  # Orange
        skeleton_info[48]["color"] = (255, 229, 0)  # Yellow

        skeleton_info[49]["color"] = (204, 255, 0)  # Lime
        skeleton_info[50]["color"] = (128, 255, 0)  # Green
        skeleton_info[51]["color"] = (51, 255, 0)  # Green
        skeleton_info[52]["color"] = (0, 255, 25)  # Green

        skeleton_info[53]["color"] = (0, 255, 102)  # Green
        skeleton_info[54]["color"] = (0, 255, 179)  # Green
        skeleton_info[55]["color"] = (0, 255, 255)  # Aqua
        skeleton_info[56]["color"] = (0, 178, 255)  # Turquoise

        skeleton_info[57]["color"] = (0, 102, 255)  # Blue
        skeleton_info[58]["color"] = (0, 25, 255)  # Blue
        skeleton_info[59]["color"] = (51, 0, 255)  # Blue
        skeleton_info[60]["color"] = (128, 0, 255)  # Purple

        skeleton_info[61]["color"] = (204, 0, 255)
        skeleton_info[62]["color"] = (255, 0, 230)
        skeleton_info[63]["color"] = (227, 0, 136)  # Pink
        skeleton_info[64]["color"] = (255, 0, 77)  # Red

        for x in [17, 15, 16]:
            with suppress(ValueError):
                ALLOWED_IDs.remove(x)
        for x in [18, 19, 20]:
            with suppress(ValueError):
                ALLOWED_IDs.remove(x)

        skeleton_info[2] = LinkInfo(link=(12, 14), id=2, color=(0, 153, 102))
        skeleton_info[7] = LinkInfo(link=((6, 5), 6), id=7, color=(255, 0, 0))
        skeleton_info[65] = LinkInfo(link=((6, 5), 0), id=65, color=(0, 0, 255))
        skeleton_info[66] = LinkInfo(link=((6, 5), 5), id=66, color=(255, 128, 0))

        skeleton_info[67] = LinkInfo(link=(10, (6, 5)), id=67, color=(0, 255, 0))
        skeleton_info[68] = LinkInfo(link=(9, (6, 5)), id=68, color=(0, 255, 255))

    if isinstance(img, torch.Tensor):
        H, W, _C = img.shape
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
    elif isinstance(img, np.ndarray):
        _C, H, W = img.shape
        img_np = img.copy()
    elif target_size is not None:
        W, H = target_size
        img_np = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        if draw_on_black:
            raise RuntimeError(
                "Cannot determine output size. Provide 'img' or 'target_size'."
            )
        else:
            raise RuntimeError("`img` must be provided if `draw_on_black` is False.")

    if draw_on_black and img is None:
        img_np = np.zeros((H, W, 3), dtype=np.uint8)

    instance_keypoints, instance_scores, keypoints_visible = results

    # pad bbox if missing
    if bboxes is None:
        instance_bboxes = [None] * len(instance_keypoints)
    else:
        instance_bboxes = [bboxes]

    for kpts, score, visible, bbox in zip(
        instance_keypoints,
        instance_scores,
        keypoints_visible,
        instance_bboxes,
        strict=True,
    ):
        if isinstance(kpts, torch.Tensor):
            kpts = np.array(kpts.float(), copy=False)

        if verbose:
            print(tensor_info(kpts, "KPTS"))

        if (
            kpt_colors is None
            or isinstance(kpt_colors, str)
            or len(kpt_colors) != len(kpts)
        ):
            raise ValueError(
                f"the length of kpt_color "
                f"({len(kpt_colors)}) does not matches "
                f"that of keypoints ({len(kpts)})"
            )

        if bitmask is not None:
            # score[..., ~bitmask] = 0
            score = np.where(bitmask, score, np.zeros_like(score))

        score = np.where(visible, score, np.zeros_like(score))
        if draw_bbox and bbox is not None:
            if len(bbox) == 0:
                log.warning("No bboxes found, skipping bbox drawing")

            else:
                for box in bbox:
                    cv2.rectangle(
                        img_np,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        bbox_color,
                        bbox_thickness,
                    )
        if thickness > 0 and draw_skeleton:
            # draw skeleton
            for skid, link_info in skeleton_info.items():
                if skid in SKIP_LINKS:
                    continue
                pt1_idx, pt2_idx = link_info["link"]

                skip_check = False
                pt1 = None
                pt2 = None

                if isinstance(pt1_idx, tuple):
                    if isinstance(pt1_idx[0], tuple):
                        _pt0 = midpoint(kpts[pt1_idx[0][0]], kpts[pt1_idx[0][1]])
                        _pt1 = midpoint(kpts[pt1_idx[1][0]], kpts[pt1_idx[1][1]])
                        pt1 = midpoint(_pt0, _pt1)
                    else:
                        pt1 = midpoint(kpts[pt1_idx[0]], kpts[pt1_idx[1]])
                    pt1_score = 1.0
                    skip_check = True

                if isinstance(pt2_idx, tuple):
                    if isinstance(pt2_idx[0], tuple):
                        _pt0 = midpoint(kpts[pt2_idx[0][0]], kpts[pt2_idx[0][1]])
                        _pt1 = midpoint(kpts[pt2_idx[1][0]], kpts[pt2_idx[1][1]])
                        pt1 = midpoint(_pt0, _pt1)

                    pt2 = midpoint(kpts[pt2_idx[0]], kpts[pt2_idx[1]])
                    pt2_score = 1.0
                    skip_check = True

                if not skip_check and (
                    pt1_idx not in ALLOWED_IDs or pt2_idx not in ALLOWED_IDs
                ):
                    continue
                color = link_info["color"]
                # [::-1]  # BGR

                if pt1 is None:
                    pt1 = kpts[pt1_idx]
                    pt1_score = score[pt1_idx]
                if pt2 is None:
                    pt2 = kpts[pt2_idx]
                    pt2_score = score[pt2_idx]

                if pt1_score > kpt_thr and pt2_score > kpt_thr:
                    x1_coord = int(pt1[0])
                    y1_coord = int(pt1[1])
                    x2_coord = int(pt2[0])
                    y2_coord = int(pt2[1])
                    cv2.line(
                        img_np,
                        (x1_coord, y1_coord),
                        (x2_coord, y2_coord),
                        color,
                        thickness=thickness,
                    )

        if radius > 0 and draw_keypoints:
            for kid, kpt in enumerate(kpts):
                if kid not in ALLOWED_IDs:
                    continue
                if score[kid] < kpt_thr:
                    if verbose:
                        log.debug(f"skipping low score: {score[kid]}/{kpt_thr}")
                    continue
                # if not visible[kid]:
                #     if verbose:
                #         log.debug("skipping not visible")
                #     continue
                if kpt_colors[kid] is None:
                    if verbose:
                        log.debug("skipping as no color")
                    continue

                color = kpt_colors[kid]

                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)

                x_coord, y_coord = int(kpt[0]), int(kpt[1])

                if verbose:
                    log.info("drawing points")
                    log.info(
                        f"Attempting to draw at ({x_coord}, {y_coord}) on an image of shape {img_np.shape} with radius {radius} and color {color}"
                    )

                if draw_labels is not None and draw_labels != "off":
                    if draw_labels == "name":
                        label = GOLIATH_KEYPOINTS[kid]
                    elif draw_labels == "id":
                        label = str(kid)
                    else:
                        label = "link"
                    cv2.putText(
                        img_np,
                        label,
                        (x_coord + radius * 2, y_coord + radius * 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_size,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                cv2.circle(img_np, (x_coord, y_coord), int(radius), color, -1)

    return img_np


# - RANDOM COMFY UTILS
def format_bytes(byte_count):
    """Formats a byte count into a human-readable string (KB, MB, GB, etc.)."""
    if byte_count is None:
        return "N/A"
    power = 1024
    n = 0
    power_labels = {0: "", 1: "K", 2: "M", 3: "G", 4: "T"}
    while byte_count >= power and n < len(power_labels) - 1:
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}B"


class TimedContext:
    def __init__(self, name: str, log_level=logging.INFO, profile=True):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.profile_memory = profile

        self.process = None
        self.mem_before = 0

    def __enter__(self):
        """Called when entering the 'with' block."""
        self.start_time = time.perf_counter()
        if self.profile_memory:
            self.process = psutil.Process(os.getpid())
            self.mem_before = self.process.memory_info().rss

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting the 'with' block."""
        if self.start_time is None:
            return False

        end_time = time.perf_counter()
        duration = end_time - self.start_time

        message = f"{self.name} executed in {duration:.4f} seconds"

        if self.profile_memory:
            mem_after = self.process.memory_info().rss
            growth = mem_after - self.mem_before

            mem_report = (
                f" | Mem: {format_bytes(mem_after)} "
                f"({'+' if growth >= 0 else ''}{format_bytes(growth)})"
            )
            message += mem_report

        if self.log_level == logging.DEBUG:
            log.debug(message)
        elif self.log_level == logging.INFO:
            log.info(message)
        elif self.log_level == logging.WARNING:
            log.warning(message)
        elif self.log_level == logging.ERROR:
            log.error(message)
        elif self.log_level == logging.CRITICAL:
            log.critical(message)
        else:
            log.info(message)

        return False


timed = TimedContext


def timed_comfy_node(task_name: str, *, profile_memory: bool = True):
    """
    A decorator to time and optionally profile memory for a node function.

    Args:
        task_name (str): The name to use as a title.
        profile_memory (bool): If True, profiles the function's memory usage.
                               Defaults to False.
    """

    # This outer function is the decorator factory
    def decorator(func):
        # functools.wraps is important to preserve the original function's metadata
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            log.info(f"Starting '{task_name}'")

            process = psutil.Process(os.getpid()) if profile_memory else None
            mem_before = process.memory_info().rss if profile_memory else 0
            _start = time.perf_counter()
            result = None

            try:
                result = func(self, *args, **kwargs)
            finally:
                _end = time.perf_counter()
                duration = _end - _start

                end_message = f"<b>{task_name}</b>: {duration:.4f} seconds"
                if profile_memory:
                    mem_after = process.memory_info().rss
                    growth = mem_after - mem_before

                    mem_report = (
                        f" | Mem: {format_bytes(mem_after)} "
                        f"({'+' if growth >= 0 else ''}{format_bytes(growth)})"
                    )
                    end_message += mem_report
                log.info(end_message)

                if (
                    hasattr(self, "unique_id")
                    and self.unique_id
                    and hasattr(self, "send_progress")
                ):
                    self.send_progress(end_message, add=True)

            return result

        return wrapper

    return decorator
