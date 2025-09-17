# ruff: noqa: F722 - Syntax error in forward annotation: {parse_error}
from functools import partial
import gc
import kornia
from pathlib import Path

from mmdet.utils.typing_utils import ConfigDict
from mmengine.dataset import Compose  # , pseudo_collate
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
# from memory_profiler import profile

from typing import Annotated
from .constants import PREPROCESS_H, PREPROCESS_MEAN, PREPROCESS_STD, PREPROCESS_W

from .datatypes import (
    BBox,
    ImageTensor,
    ImageTensorBatch,
    ImageTensorBatchChannelFirst,
    KeypointResult,
    ProgressCallable,
    SapiensResult,
    UInt8,
    Float,
)

from .proxy_tensor import LazyProxyTensor, is_tensor, tensor_info

from .utils import (
    chunked_tensors,
    log,
    pad_tensor_to_batch_size,
    timed,
    to_np,
)


from typing import Callable, Sequence
from mmengine.structures import BaseDataElement
import rich
import torch
import numpy as np
import math
import cv2
import torch.nn as nn
import contextlib
import io


def progress_fallback(*, it=None, msg=None, title=None):
    if it:
        log.info(f"Processed {it} iterations")
        return

    if msg and title:
        log.info(f"{title}: {msg}")
        return

    if msg:
        log.info(msg)
        return


ImagesType = np.ndarray | Sequence[np.ndarray]


def load_mmdet(model_path, *, device: str = "cuda"):
    root = Path(__file__).parent.parent
    mmdet_config = root / "configs" / "rtmdet_m_640-8xb32_coco-person_no_nms.py"

    _std_err = io.StringIO()
    with contextlib.redirect_stderr(_std_err):
        detector = init_detector(mmdet_config.as_posix(), model_path, device=device)

    if isinstance(detector.cfg, ConfigDict):
        if "test_dataloader" not in detector.cfg:
            return detector

        pipeline = detector.cfg.test_dataloader.dataset.pipeline
        for trans in pipeline:
            if trans["type"] in dir(transforms):
                trans["type"] = "mmdet." + trans["type"]

    return detector


def _inference_detector(
    model: torch.nn.Module,
    imgs: torch.Tensor,  # ImagesType,
    test_pipeline: Compose | ConfigDict | None = None,
    text_prompt: str | None = None,
    custom_entities: bool = False,
) -> DetDataSample | SampleList:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    from mmengine.dataset import Compose


    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], (np.ndarray, torch.Tensor)):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = "mmdet.LoadImageFromNDArray"

        # test_pipeline = test_pipeline[1:]

        test_pipeline = Compose(test_pipeline)


    result_list = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        elif isinstance(img, torch.Tensor):
            mult = 255 if img.dtype == torch.float32 else 1
            data_ = dict(img=(img * mult).numpy(), img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_["text"] = text_prompt
            data_["custom_entities"] = custom_entities

        # build the data pipeline
        data_ = test_pipeline(data_)

        data_["inputs"] = [data_["inputs"]]
        data_["data_samples"] = [data_["data_samples"]]

        # forward the model
        with torch.no_grad():
            results = model.test_step(data_)[0]

        result_list.append(results)

    return result_list


def _nms(dets: np.ndarray, thr: float):
    """Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets (np.ndarray): [[x1, y1, x2, y2, score]].
        thr (float): Retain overlap < thr.

    Returns:
        list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep


def _process_one_image_bbox(
    pred_instance,
    *,
    det_cat_id: int = 0,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    bbox_padding: float = 0.0,
) -> BBox:
    """
    Processes raw detections for a single image to get final bounding boxes.
    Includes filtering, NMS, and optional padding.
    """
    valid_indices = np.logical_and(
        pred_instance.labels == det_cat_id,
        pred_instance.scores > bbox_thr,
    )

    bboxes_with_scores = np.concatenate(
        (
            pred_instance.bboxes[valid_indices],
            pred_instance.scores[valid_indices, None],
        ),
        axis=1,
    )

    if bboxes_with_scores.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    keep_indices = _nms(bboxes_with_scores, nms_thr)
    final_bboxes = bboxes_with_scores[keep_indices, :4]

    if bbox_padding > 0.0 and final_bboxes.shape[0] > 0:
        x1, y1, x2, y2 = final_bboxes.T
        widths = x2 - x1
        heights = y2 - y1
        if bbox_padding <= 1.0:
            # percentage
            pad_x = widths * bbox_padding
            pad_y = heights * bbox_padding
        else:
            # absolute
            pad_x = bbox_padding
            pad_y = bbox_padding

        final_bboxes[:, 0] = x1 - pad_x / 2
        final_bboxes[:, 1] = y1 - pad_y / 2
        final_bboxes[:, 2] = x2 + pad_x / 2
        final_bboxes[:, 3] = y2 + pad_y / 2

    return final_bboxes
# @jaxtyped(typechecker=typechecker)
def extract_bboxes(
    imgs: ImageTensorBatch,
    detector: nn.Module,
    *,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    bbox_padding: float = 0.0,
    chunk_size: int = 16,
) -> list[BBox]:
    from tqdm import tqdm

    if not isinstance(imgs, torch.Tensor | LazyProxyTensor):
        raise TypeError(f"Expected input to be a torch.Tensor, but got {type(imgs)}")

    all_det_results = []
    log.info(
        f"[BBox Detector] Processing {len(imgs)} frames in chunks of {chunk_size}..."
    )

    for i in tqdm(range(0, len(imgs), chunk_size), desc="Detecting Bounding Boxes"):
        # get chunk and swap channels for cv2
        chunk_tensor = imgs[i : i + chunk_size][..., [2, 1, 0]]
        with torch.no_grad():
            det_results_chunk = _inference_detector(detector, chunk_tensor)

        all_det_results.extend(det_results_chunk)

        del chunk_tensor, det_results_chunk  # , chunk_np,
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("BBox inference complete.")

    # predict bbox

    def extract_pred(r: DetDataSample) -> BaseDataElement:
        return r.pred_instances.cpu().numpy()

    with timed("Bbox post-processing"):
        pred_instances = list(
            map(
                extract_pred,
                all_det_results,
            )
        )
        process_func = partial(
            _process_one_image_bbox,
            det_cat_id=0,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            bbox_padding=bbox_padding,  # Pass the padding value down
        )
        bboxes_batch = list(map(process_func, pred_instances))

    return bboxes_batch


def _top_down_affine_transform(img, bbox, padding=1.25):
    """
    Args:
        img (np.ndarray): Image to be transformed.
        bbox (np.ndarray): Bounding box to be transformed.
        padding (int): Padding size.

    Returns:
        np.ndarray: Transformed image.
        np.ndarray: Transformed bounding box.
    """
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    h, w = img.shape[:2]
    warp_size = (int(w), int(h))
    aspect_ratio = w / h

    # reshape bbox to fixed aspect ratio
    box_w, box_h = np.hsplit(scale, [1])
    scale = np.where(
        box_w > box_h * aspect_ratio,
        np.hstack([box_w, box_w / aspect_ratio]),
        np.hstack([box_h * aspect_ratio, box_h]),
    )

    rot = 0.0

    warp_mat = _get_udp_warp_matrix(center, scale, rot, output_size=(w, h))

    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, [center], [scale]


# @jaxtyped(typechecker=typechecker)
def preprocess_pose(
    orig_img: ImageTensor,
    bboxes: BBox,
    *,
    # only customisable for models using Dim.DYNAMIC for H and W
    input_shape: tuple[int, int] = (
        PREPROCESS_H,
        PREPROCESS_W,
    ),
    mean: tuple[float, float, float] = PREPROCESS_MEAN,
    std: tuple[float, float, float] = PREPROCESS_STD,
    on_update: Callable[[int], None] | None = None,
) -> tuple[
    # cropped images
    list[Float[torch.Tensor, "C W_out H_out"]],
    # centers
    list[Float[np.ndarray, "2"]],
    # scales
    list[Float[np.ndarray, "2"]],
]:
    """Crops and preprocess the orig image using the data from bboxes"""
    preprocessed_images: list[torch.Tensor] = []
    centers: list[np.ndarray] = []
    scales: list[np.ndarray] = []

    if is_tensor(orig_img):
        img_np = to_np(orig_img)
    else:
        img_np = orig_img

    for bbox in bboxes:
        img, center, scale = _top_down_affine_transform(img_np.copy(), bbox)
        img = cv2.resize(
            img,
            (input_shape[1], input_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        mean_tensor = torch.Tensor(mean).view(-1, 1, 1)
        std_tensor = torch.Tensor(std).view(-1, 1, 1)
        img = (img - mean_tensor) / std_tensor
        preprocessed_images.append(img)
        centers.extend(center)
        scales.extend(scale)
        if on_update is not None:
            # on_update(len(preprocessed_images))
            on_update(1)

    return preprocessed_images, centers, scales


def _get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size,
) -> np.ndarray:
    """Calculate the affine transformation matrix under the unbiased
    constraint. See `UDP (CVPR 2020)`_ for details.

    Note:

        - The bbox number: N

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (tuple): Size ([w, h]) of the output image

    Returns:
        np.ndarray: A 2x3 transformation matrix

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (
        -0.5 * input_size[0] * math.cos(rot_rad)
        + 0.5 * input_size[1] * math.sin(rot_rad)
        + 0.5 * scale[0]
    )
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (
        -0.5 * input_size[0] * math.sin(rot_rad)
        - 0.5 * input_size[1] * math.cos(rot_rad)
        + 0.5 * scale[1]
    )
    return warp_mat


# @jaxtyped(typechecker=typechecker)


def batch_inference_topdown(
    model: Annotated[nn.Module, "Sapiens Model"],
    imgs: ImageTensorBatchChannelFirst,  # Float[torch.Tensor, "B 3 H W"],
    # imgs: list[np.ndarray | str],
    dtype=torch.bfloat16,
    flip=False,
    chunk_size: int = 8,
) -> ImageTensorBatchChannelFirst:
    # Float[torch.Tensor, "B C_out H_out W_out"]:
    B_full = imgs.shape[0]  # Total number of images in the input batch

    n_chunks = (B_full + chunk_size - 1) // chunk_size

    all_heatmaps: list[torch.Tensor] = []

    for _i in range(n_chunks):
        start_idx = _i * chunk_size
        end_idx = min((_i + 1) * chunk_size, B_full)

        current_img_slice = imgs[start_idx:end_idx]
        valid_len = current_img_slice.shape[0]

        # padded_img_chunk = pad_images_to_batchsize(current_img_slice, chunk_size)
        padded_img_chunk = pad_tensor_to_batch_size(current_img_slice, chunk_size)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            # This explicitly signals the start of a cudagraph step.
            # It's important for models compiled with `torch.compile` that use graph capture.
            torch.compiler.cudagraph_mark_step_begin()

            img_chunk_gpu = padded_img_chunk.to(dtype).cuda()

            current_heatmaps_gpu = model(img_chunk_gpu)

            if flip:
                flipped_heatmaps_gpu = model(img_chunk_gpu.flip(-1))
                current_heatmaps_gpu = (
                    current_heatmaps_gpu + flipped_heatmaps_gpu
                ) * 0.5

            all_heatmaps.append(current_heatmaps_gpu[:valid_len].cpu())

            del current_heatmaps_gpu
            del img_chunk_gpu
            # TODO: conditionally call that?
            torch.cuda.empty_cache()
            gc.collect()

    final_heatmaps = torch.cat(all_heatmaps, dim=0)

    return final_heatmaps

    # with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
    #     heatmaps = model(imgs.cuda())
    #     if flip:
    #         heatmaps_ = model(imgs.to(dtype).cuda().flip(-1))
    #         heatmaps = (heatmaps + heatmaps_) * 0.5
    #     imgs = imgs.cpu()
    # return heatmaps.cpuk()


# @jaxtyped(typechecker=typechecker)


def _udp_decode(
    heatmaps: Float[
        np.ndarray, "308 256 192"
        # "N 308 256 192"
    ],  #: BFloat16[torch.Tensor, "B N 308 256 192"],
    input_shape: tuple[int, int],
    heatmap_size: tuple[int, int],
    blur_kernel_size: int = 11,
    verbose=False,
) -> tuple[np.ndarray, np.ndarray]:
    """UDP decoding for keypoint location refinement.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        blur_kernel_size (int): Gaussian kernel size (K) for modulation, which
            should match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Refined keypoint locations.
    """
    keypoints, scores = _get_heatmap_maximum(heatmaps)
    if verbose:
        rich.print("Keypoints[0]:")
        rich.print(keypoints[0])

        rich.print("Scores[0]:")
        rich.print(scores[0])

    # unsqueeze the instance dimension for single-instance results
    keypoints = keypoints[None]
    scores = scores[None]

    keypoints = _refine_keypoints_dark_udp(
        keypoints, heatmaps, blur_kernel_size=blur_kernel_size
    )

    W, H = heatmap_size
    keypoints = (keypoints / [W - 1, H - 1]) * input_shape
    return keypoints, scores


def _gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """

    if True:
        K, H, W = heatmaps.shape
        heatmaps_out = np.empty_like(heatmaps)

        sigmaX = 0

        for k in range(K):
            origin_max = np.max(heatmaps[k])

            # Apply Gaussian blur directly to the heatmap slice.
            # borderType=cv2.BORDER_CONSTANT and value=0 mimics the zero-padding behavior
            # of your original `dr = np.zeros(...)` logic, but much more efficiently
            # within OpenCV's optimized kernel.
            blurred_slice = cv2.GaussianBlur(
                heatmaps[k],
                (kernel, kernel),
                sigmaX,
                borderType=cv2.BORDER_CONSTANT,
                dst=heatmaps_out[
                    k
                ],  # Optional: Write directly to the output slice for efficiency
            )

            # Check if blurred_slice is actually heatmaps_out[k] (it should be if dst was used)
            # If dst was not used, or if the function returns a new array, ensure assignment:
            if not np.shares_memory(blurred_slice, heatmaps_out[k]):
                heatmaps_out[k] = blurred_slice  # Assign if a new array was returned

            current_max = np.max(heatmaps_out[k])  # Max of the *blurred* slice

            # Rescale the blurred heatmap to match original max
            if current_max > 0:
                heatmaps_out[k] *= origin_max / current_max
            elif origin_max == 0:
                pass  # Already all zeros
            else:  # origin_max > 0 but current_max == 0
                # This means the blur completely "flattened" the heatmap to zero.
                # We want to ensure it remains zero, not infinity or NaN.
                heatmaps_out[k].fill(0)  # Explicitly set to zero

        return heatmaps_out

    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k]
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border]
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def _get_heatmap_maximum_torch(
    heatmaps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get maximum response location and value from heatmaps using PyTorch.

    This is the idiomatic "batched" version which is more efficient for
    GPU pipelines as it avoids creating a Python list of tensors.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (torch.Tensor): Heatmaps tensor of shape (B, K, H, W)
                                 on the target device (e.g., CUDA).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - locs (torch.Tensor): Locations of maximums for the entire batch,
            shape (B, K, 2) in (x, y) format.
        - vals (torch.Tensor): Values of maximums for the entire batch,
            shape (B, K).
    """
    # --- Input validation ---
    assert isinstance(heatmaps, torch.Tensor), "heatmaps should be a torch.Tensor"
    assert heatmaps.ndim == 4, (
        f"Heatmaps tensor must be 4D, but got shape {heatmaps.shape}"
    )

    # --- Get tensor dimensions ---
    B, K, H, W = heatmaps.shape

    # --- Find the maximum value and its flat index in each HxW heatmap ---
    # 1. Flatten the last two dimensions (H, W) into a single dimension of size H*W.
    #    The view method is memory-efficient as it doesn't copy data.
    #    Shape changes from (B, K, H, W) -> (B, K, H*W).
    heatmaps_flat = heatmaps.view(B, K, -1)

    # 2. Find the max value and its index along the last dimension (dim=2).
    #    `torch.max` returns a named tuple, but we unpack it into two tensors.
    vals, indices = torch.max(heatmaps_flat, dim=2)
    # `vals` shape: (B, K). Contains the max value for each of the B*K heatmaps.
    # `indices` shape: (B, K). Contains the *flat* index (from 0 to H*W-1) of the max value.

    # --- Convert the flat index back to 2D (x, y) coordinates ---
    # This is the PyTorch equivalent of np.unravel_index.
    # 3. Calculate the x-coordinate (column) using the modulo operator.
    x_locs = indices % W

    # 4. Calculate the y-coordinate (row) using integer division.
    y_locs = indices // W

    # 5. Stack the x and y coordinates to form the final location tensor.
    #    We stack along a new last dimension (dim=2).
    #    `x_locs` and `y_locs` have shape (B, K), so the result has shape (B, K, 2).
    #    We also cast to float, as coordinates are often expected as floats.
    locs = torch.stack([x_locs, y_locs], dim=2).to(torch.float32)

    # --- Handle invalid locations (where max value <= 0) ---
    # 6. Create a boolean mask where the condition is true.
    #    `mask` will have shape (B, K).
    mask = vals <= 0.0

    # 7. Use the mask to set the corresponding locations to -1.
    #    PyTorch will broadcast the (B, K) mask to the (B, K, 2) locs tensor,
    #    setting both the x and y coordinates to -1 where the condition is met.
    locs[mask] = -1.0

    return locs, vals


def _get_heatmap_maximum(heatmaps: np.ndarray):
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, f"Invalid shape {heatmaps.shape}"

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.0] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def _refine_keypoints_dark_udp(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int
) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP. See `UDP`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = _gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum("imn,ink->imk", hessian, derivative).squeeze()

    return keypoints


# @jaxtyped(typechecker=typechecker)


def extract_keypoints(
    result: SapiensResult,
    *,
    input_shape: tuple[int, int] = (PREPROCESS_W, PREPROCESS_H),
    heatmap_scale: int = 4,
    # *,
    verbose=False,
    on_update: Callable[[int], None] | None = None,
    udp_blur=11,
) -> KeypointResult:
    # pred_instances_list = split_instances(result)

    with timed("udp decode"):
        instance_keypoints = []
        instance_scores = []
        for i in range(len(result.heatmaps)):
            if verbose:
                print(tensor_info(result.heatmaps[i], "udp prep before"))
            t = result.heatmaps[i].cpu().unsqueeze(0).float()
            if verbose:
                print(tensor_info(t, "udp prep after"))

            decoded = _udp_decode(
                t[0].numpy(),
                input_shape,
                (
                    int(input_shape[0] / heatmap_scale),
                    int(input_shape[1] / heatmap_scale),
                ),
                udp_blur,
                verbose=verbose,
            )

            keypoints, keypoint_scores = decoded

            if verbose:
                print("SHAPE DECODED KP", keypoints.shape)
                print("SHAPE DECODED SCORE", keypoint_scores.shape)

            keypoints = (
                (keypoints / input_shape) * result.scales[i]
                + result.centres[i]
                - 0.5 * result.scales[i]
            )
            instance_keypoints.append(keypoints[0])
            instance_scores.append(keypoint_scores[0])
            if on_update is not None:
                on_update(1)

        instance_keypoints = np.array(instance_keypoints).astype(np.float32)
        instance_scores = np.array(instance_scores).astype(np.float32)

        np.max(instance_scores)
        keypoints_visible = np.ones(instance_keypoints.shape[:-1], dtype=np.float32)

        return KeypointResult(
            keypoints=instance_keypoints,
            scores=instance_scores,
            visible=keypoints_visible,
        )


def preprocess_bboxes(
    frames: UInt8[torch.Tensor, "B H W C"],
    bboxes: Float[torch.Tensor, "B N 4"],
    *,
    input_shape: tuple[int, int] = (PREPROCESS_H, PREPROCESS_W),
    mean: tuple[float, float, float] = PREPROCESS_MEAN,
    std: tuple[float, float, float] = PREPROCESS_STD,
) -> tuple[
    # cropped, resized, and normalized images
    Float[torch.Tensor, "B N C_out H_out W_out"],
    # centers
    Float[torch.Tensor, "B N 2"],
    # scales
    Float[torch.Tensor, "B N 2"],
]:
    """
    Preprocesses a batch of frames and bounding boxes for pose estimation using vectorized torch operations.
    This version is robust to padded (all-zero) bounding boxes.
    """
    B, N, _ = bboxes.shape
    _, H, W, C = frames.shape
    out_h, out_w = input_shape
    device = frames.device

    # --- 1. Create a mask for valid (non-padded) boxes ---
    # A valid box has a positive width (x2 > x1). This safely handles [0,0,0,0].
    valid_mask = bboxes[..., 2] > bboxes[..., 0]  # Shape: (B, N)

    # --- 2. Calculate center and scale from bboxes ---
    x1, y1, x2, y2 = bboxes.unbind(dim=-1)
    # Add a small epsilon to width and height to prevent division by zero for valid but zero-area boxes.
    w = x2 - x1 + 1e-6
    h = y2 - y1 + 1e-6

    centers = torch.stack([x1 + w * 0.5, y1 + h * 0.5], dim=-1)

    # --- 3. Adjust scale to match the target aspect ratio ---
    aspect_ratio = out_w / out_h
    padding_factor = 1.25

    is_wider = w / h > aspect_ratio
    h_new = torch.where(is_wider, w / aspect_ratio, h)
    w_new = torch.where(~is_wider, h * aspect_ratio, w)

    scales_for_transform = torch.stack(
        [w_new * padding_factor, h_new * padding_factor], dim=-1
    )
    scales_for_return = torch.stack([w_new, h_new], dim=-1) / 200.0

    # --- 4. Prepare tensors for kornia.warp_affine ---
    frames_bchw = frames.permute(0, 3, 1, 2).float()
    frames_expanded = (
        frames_bchw.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)
    )

    centers_flat = centers.view(B * N, 2)
    scales_flat = scales_for_transform.view(B * N, 2)

    # --- 5. Construct the 2x3 affine transformation matrix ---
    M = torch.zeros(B * N, 2, 3, device=device, dtype=torch.float32)
    M[:, 0, 0] = scales_flat[:, 0] / out_w
    M[:, 1, 1] = scales_flat[:, 1] / out_h
    M[:, 0, 2] = centers_flat[:, 0] - scales_flat[:, 0] / 2
    M[:, 1, 2] = centers_flat[:, 1] - scales_flat[:, 1] / 2

    # --- KEY CHANGE: Handle singular matrices for padded boxes ---
    # Create a safe, non-singular placeholder matrix (identity transform)
    # This matrix will be used for any bbox that is marked as invalid.
    identity_matrix = torch.eye(2, 3, device=device).expand(B * N, -1, -1)

    # Reshape mask to be broadcastable with the matrix M
    mask_for_M = valid_mask.view(B * N, 1, 1)

    # Use torch.where to select the real matrix for valid boxes and the identity for invalid ones
    final_M = torch.where(mask_for_M, M, identity_matrix)
    # --- END OF KEY CHANGE ---

    # --- 6. Perform the batched affine warp ---
    # This call is now safe because `final_M` contains no singular matrices.
    cropped_images = kornia.geometry.transform.warp_affine(
        frames_expanded,
        final_M,
        dsize=(out_h, out_w),
        mode="bilinear",
        padding_mode="zeros",
    )

    # --- 7. Post-process the crops ---
    crops = cropped_images.view(B, N, C, out_h, out_w)
    crops = crops[:, :, [2, 1, 0], :, :]  # BGR -> RGB

    mean_t = torch.tensor(mean, device=device).view(1, 1, C, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 1, C, 1, 1)
    normalized_crops = (crops - mean_t) / std_t

    # --- 8. Mask out ALL results for invalid/padded instances ---
    # This is crucial. Even though we used an identity matrix, the output for
    # padded boxes is meaningless. We must zero it out to ensure correctness.
    valid_mask_expanded = valid_mask.view(B, N, 1, 1, 1)
    normalized_crops = normalized_crops * valid_mask_expanded

    valid_mask_2d = valid_mask.unsqueeze(-1)
    centers = centers * valid_mask_2d
    scales_for_return = scales_for_return * valid_mask_2d

    return normalized_crops, centers, scales_for_return


def preprocess(
    detector: nn.Module,
    image: torch.Tensor,
    *,
    confidence_threshold: float = 0.3,
    nms_threshold: float = 0.3,
    bbox_padding: float = 0.1,
    on_update: ProgressCallable | None = None,
):
    on_update = on_update or progress_fallback
    bboxes_batch = extract_bboxes(
        # channel swapping for cv2
        image.clone()[..., [2, 1, 0]],
        detector,
        bbox_thr=confidence_threshold,
        nms_thr=nms_threshold,
    )
    on_update(it=1)
    _missed_bbox = 0
    _total_bboxes = 0
    img_bbox_map = {}

    # fill missing bbox
    for _i, _bboxes in enumerate(bboxes_batch):
        if len(_bboxes) == 0:
            _missed_bbox += 1
            bboxes_batch[_i] = np.array(
                [[0, 0, image.shape[1], image.shape[2]]], dtype=np.float32
            )

        _bbox_count = len(bboxes_batch[_i])
        img_bbox_map[_i] = _bbox_count
        _total_bboxes += _bbox_count
    if _missed_bbox:
        on_update(
            msg=f"BBox not found for {_missed_bbox} frames / {image.shape[0]} frames"
        )
    on_update(
        msg=f"Found {_total_bboxes} bboxes for {image.shape[0] - _missed_bbox} / {image.shape[0]} frames.",
    )

    on_update(it=1)

    assert len(bboxes_batch) == image.shape[0], "Unexpected mismatch of bbox & frames"
    return bboxes_batch


def estimate_pose(
    sapiens: nn.Module,
    frames: torch.Tensor,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    image: Float[torch.Tensor, "B H W C"] | None = None,
    chunk_size=16,
    on_update: ProgressCallable | None = None,
):
    # inference
    torch.compiler.cudagraph_mark_step_begin()
    pose_imgs, pose_img_centers, pose_img_scales = data
    n_pose_batches = (len(pose_imgs) + chunk_size - 1) // chunk_size
    pose_results: list[Float[torch.Tensor, "B C_out H_out W_out"]] = []

    on_update = on_update or progress_fallback

    from dataclasses import dataclass

    @dataclass
    class Data:
        crops: torch.Tensor
        centres: torch.Tensor
        scales: torch.Tensor

    # - main sapien inference
    with torch.no_grad():
        c = 0
        for t, length in chunked_tensors(
            Data, crops=pose_imgs, centres=pose_img_centers, scales=pose_img_scales
        ):
            for frame in t.crops[:length]:
                pose_results.extend(batch_inference_topdown(sapiens, frame))
                on_update(it=1)
                on_update(
                    msg=f"Processing chunk {c + 1}/{n_pose_batches}",  # for _i in range(n_pose_batches):
                    title="Pose Estimation",  #     imgs = torch.stack(
                )  #         pose_imgs[_i * chunk_size : (_i + 1) * chunk_size], dim=0
                c += 1
        #     )
        #     valid_len = len(imgs)
        #     imgs = pad_tensor_to_batch_size(imgs, chunk_size)
        #
        #     pose_results.extend(
        #         batch_inference_topdown(sapiens, imgs, dtype=torch.bfloat16)[:valid_len]
        #     )
        #     del imgs
        #     on_update(it=1)
        #     # pbar.update(1)
        #     on_update(
        #         msg=f"Processing chunk {_i + 1}/{n_pose_batches}",
        #         title="Pose Estimation",
        #     )
        #
    return pose_results


__all__ = [
    # PRIVATE
    # "_inference_detector",
    # "_process_one_image_bbox",
    # "_nms",
    # "_top_down_affine_transform",
    # "_get_udp_warp_matrix",
    # "_udp_decode",
    # "_gaussian_blur",
    # "_get_heatmap_maximum",
    # "_refine_keypoints_dark_udp",
    # PUBLIC
    "extract_bboxes",
    "preprocess_pose",
    "batch_inference_topdown",
    # RE-EXPORT
    # "init_detector",
    # "get_device",
    # "init_default_scope",
]
