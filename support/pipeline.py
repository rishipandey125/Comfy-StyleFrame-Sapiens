# ruff: noqa: F722 - Syntax error in forward annotation: {parse_error}
import contextlib
from functools import lru_cache
import gc
from typing import Literal
from scipy.signal import savgol_filter
import torch
from pathlib import Path
import numpy as np

from .io import add_metadata

from .classes_and_palettes import (
    GOLIATH_KEYPOINTS,
)


from .utils import calculate_proportional_radius, timed, to_np

from .proxy_tensor import LazyProxyTensor

from .yolo import Detector

from .constants import DETECTORS
from .datatypes import (
    BBox,
    BBoxPerFrame,
    ImageTensorBatch,
    KeypointResult,
    Result,
    Models,
    ProgressCallable,
    SapiensResult,
    UInt8,
)
import logging


# NOTE: runtime typechecks


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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


def _download(
    url, filename, *, comfy_progress=True, on_progress: ProgressCallable | None = None
):
    import requests
    from tqdm import tqdm

    if on_progress is None:
        on_progress = progress_fallback

    Path(filename).parent.mkdir(exist_ok=True, parents=True)

    with open(filename, "wb") as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            tqdm_params = {
                "total": total,
                "miniters": 1,
                "unit": "B",
                "unit_scale": True,
                "unit_divisor": 1024,
            }
            if comfy_progress:
                import comfy.utils

                pbar = comfy.utils.ProgressBar(total)

            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    if comfy_progress:
                        pbar.update(len(chunk))

                    on_progress(msg=f"Downloading {pb.n / pb.total * 100:.2f}%")
                    f.write(chunk)


def _download_model(
    meta,
    *,
    base_path: Path | None = None,
    auto_download=False,
    on_progress: ProgressCallable | None = None,
    comfy_progress=False,
):
    if base_path is None:
        import folder_paths

        path = folder_paths.get_full_path_or_raise(
            "sapiens", f"{meta['name']}/{meta['filename']}"
        )
        path = Path(path)
    else:
        path = base_path / meta["name"] / meta["filename"]

    if on_progress is None:
        on_progress = progress_fallback

    if path.exists():
        on_progress(msg="Model found locally.")
        return path.as_posix()

    if auto_download:
        from huggingface_hub import hf_hub_url

        on_progress(msg=f"Model {path.name} not found, downloading from HuggingFace...")

        repo_id = meta["repo"]

        url = hf_hub_url(repo_id=repo_id, filename=path.name)
        _download(
            url, path.as_posix(), comfy_progress=comfy_progress, on_progress=on_progress
        )
        log.info("Model downloaded successfully to", path)

        return path.as_posix()

    else:
        raise RuntimeError(f"Model not found at {path} and auto download is False")


def load_model(
    size="1B",
    compile=True,
    use_torchscript=False,
    device="cuda",
    bbox_backend="auto",
    auto_download=False,
    comfy_progress=False,
    on_progress: ProgressCallable | None = None,
    base_path: Path | None = None,
) -> Models:
    if on_progress is None:
        on_progress = progress_fallback

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    mode = "torchscript" if use_torchscript else "bfloat16"
    meta = {
        "repo": f"facebook/sapiens-pose-1b-{mode}"
        if use_torchscript
        else "melmass/sapiens",
        "name": "pose",
        "filename": f"sapiens_1b_goliath_best_goliath_AP_639_{mode}.pt2",
        "use_torchscript": use_torchscript,
    }

    model_path = _download_model(
        meta,
        auto_download=auto_download,
        on_progress=on_progress,
        comfy_progress=comfy_progress,
        base_path=base_path,
    )
    if compile:
        with contextlib.suppress(Exception):
            torch._inductor.config.force_fuse_int_mm_with_mul = True
            torch._inductor.config.use_mixed_mm = True

    if use_torchscript:
        dtype = torch.float32
        model = torch.jit.load(model_path).to(device).to(dtype)
        log.info("Loaded torchscript model")
    else:
        dtype = torch.bfloat16
        model = torch.export.load(model_path).module().to(device).to(dtype)
        log.info("Loaded exported program model")

    if compile:
        model = torch.compile(model, mode="default", fullgraph=True)
        # TODO: warmup chunk?

    if bbox_backend == "Yolov8":
        log.debug("Using Yolov8")
        detector = Detector()
    else:
        log.debug("Using RtmDet")
        # https://huggingface.co/facebook/sapiens-pose-bbox-detector/blob/main/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
        detector_path = _download_model(
            DETECTORS["mmdet"],
            auto_download=auto_download,
            on_progress=on_progress,
            comfy_progress=comfy_progress,
            base_path=base_path,
        )
        from .sapiens_utils import load_mmdet

        detector = load_mmdet(
            model_path=detector_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    return (model, detector)


# @jaxtyped(typechecker=typechecker)
def preprocess(
    image: ImageTensorBatch,
    *,
    model: Models | None = None,
    confidence_threshold=0.3,
    nms_threshold=0.3,
    bbox_padding=0.1,
    on_progress: ProgressCallable | None = None,
) -> BBoxPerFrame:
    """Extract BBoxes from the given frames."""
    if on_progress is None:
        on_progress = progress_fallback

    if model is None:
        model = load_model()

    _, detector = model

    if isinstance(detector, torch.nn.Module):
        from .sapiens_utils import extract_bboxes
        import numpy as np

        bboxes_batch = extract_bboxes(
            image,
            detector,
            bbox_thr=confidence_threshold,
            nms_thr=nms_threshold,
            bbox_padding=bbox_padding,
        )

        _missed_bbox = 0
        _total_bboxes = 0
        img_bbox_map = {}

        # fill missing bbox frames
        for i, _bboxes in enumerate(bboxes_batch):
            if len(_bboxes) == 0:
                _missed_bbox += 1
                h, w = image.shape[1], image.shape[2]
                bboxes_batch[i] = np.array([[0, 0, w, h]], dtype=np.float32)

            _bbox_count = len(bboxes_batch[i])
            img_bbox_map[i] = _bbox_count
            _total_bboxes += _bbox_count

        if _missed_bbox:
            on_progress(
                msg=f"BBox not found for {_missed_bbox} frames / {image.shape[0]} frames"
            )

        on_progress(
            msg=f"Found {_total_bboxes} bboxes for {image.shape[0] - _missed_bbox} / {image.shape[0]} frames.",
        )
        # return (bboxes_batch, img_bbox_map)
        return bboxes_batch

    else:
        raise ValueError("YOLO todo!")
    # self.send_progress("Using Yolo")
    #
    # # pbar = comfy.utils.ProgressBar(image.shape[0])
    #
    # # def on_update(v):
    # #     pbar.update(v)
    #
    # with torch.no_grad():
    #     all_people_bboxes, all_cropped_people = detect_and_crop_people(
    #         image,
    #         chunk_size=16,
    #         progress_callback=on_update,
    #         confidence=confidence_threshold,
    #         padding=bbox_padding,
    #     )
    #
    # if all_people_bboxes.shape[0] == 0:
    #     raise ValueError("No people found in the video.")
    #
    # return ((all_cropped_people, all_people_bboxes),)


def _create_empty_keypoint_result() -> KeypointResult:
    import numpy as np

    return KeypointResult(
        keypoints=np.empty((0, 308, 2), dtype=np.float32),
        scores=np.empty((0, 308), dtype=np.float32),
        visible=np.empty((0, 308), dtype=np.float32),
    )


# @jaxtyped(typechecker=typechecker)
def estimate_pose(
    model: Models,
    image: torch.Tensor | LazyProxyTensor,
    bboxes_frames: BBoxPerFrame,
    chunk_size=16,
    # bbox_backend="auto",
    comfy_progress=False,
    use_torch=False,  # for point extraction
    udp_blur=11,
    on_progress: ProgressCallable | None = None,
) -> Result:
    if on_progress is None:
        on_progress = progress_fallback

    (sapiens, _) = model

    # if bbox_backend == "auto" or bbox_backend == "RtmDet":
    from .utils import pad_tensor_to_batch_size
    from .sapiens_utils import batch_inference_topdown, preprocess_pose
    from .datatypes import SapiensResult
    import numpy as np
    from tqdm import tqdm

    torch.compiler.cudagraph_mark_step_begin()

    B, H, W, _C = image.shape
    final_results_by_frame = [
        {"keypoints": [], "scores": [], "visible": []} for _ in range(B)
    ]

    final_results: list[KeypointResult] = []
    batch_pose_imgs = []
    batch_pose_img_centers = []
    batch_pose_img_scales = []
    batch_frame_indices = []

    on_progress(msg="Compiling Model (if first run)...")

    if comfy_progress:
        import comfy.utils

        pbar = comfy.utils.ProgressBar(B)

    qbar = tqdm(range(B), desc="Estimating Pose", colour="yellow")
    for i in qbar:
        frame_tensor = image[i]

        frame_np_uint8 = to_np(
            frame_tensor
        )  # (frame_tensor.cpu().numpy() * dtype_mult).astype(np.uint8)

        bbox_list = bboxes_frames[i]

        op_result = preprocess_pose(frame_np_uint8, bbox_list)

        if op_result:
            person_images, centers, scales = op_result
            batch_pose_imgs.extend(person_images)
            batch_pose_img_centers.extend(centers)
            batch_pose_img_scales.extend(scales)
            batch_frame_indices.extend([i] * len(person_images))

        # process full "chunk_size" chunk
        while len(batch_pose_imgs) >= chunk_size:
            # pick first "chunk_size"

            qbar.set_description(f"Processing chunk! {i}")

            process_imgs = batch_pose_imgs[:chunk_size]
            process_centers = batch_pose_img_centers[:chunk_size]
            process_scales = batch_pose_img_scales[:chunk_size]
            process_frame_indices = batch_frame_indices[:chunk_size]

            # remove from the front of the list
            batch_pose_imgs = batch_pose_imgs[chunk_size:]
            batch_pose_img_centers = batch_pose_img_centers[chunk_size:]
            batch_pose_img_scales = batch_pose_img_scales[chunk_size:]
            batch_frame_indices = batch_frame_indices[chunk_size:]

            # process chunk
            with torch.no_grad():
                imgs_tensor = torch.stack(process_imgs, dim=0)
                valid_len = len(imgs_tensor)
                padded_imgs = pad_tensor_to_batch_size(imgs_tensor, chunk_size)

                heatmaps = batch_inference_topdown(
                    sapiens, padded_imgs, dtype=torch.bfloat16
                )[:valid_len]

            result_wrapper = SapiensResult(
                heatmaps=heatmaps,
                centres=np.stack(process_centers, axis=0).astype(np.float32),
                scales=np.stack(process_scales, axis=0).astype(np.float32),
            )

            keypoint_chunk_result = _extract_keypoints(
                result_wrapper, use_torch=use_torch, udp_blur=udp_blur
            )

            for person_idx_in_chunk in range(valid_len):
                original_frame_idx = process_frame_indices[person_idx_in_chunk]

                acc = final_results_by_frame[original_frame_idx]
                acc["keypoints"].append(
                    keypoint_chunk_result.keypoints[person_idx_in_chunk]
                )
                acc["scores"].append(keypoint_chunk_result.scores[person_idx_in_chunk])
                acc["visible"].append(
                    keypoint_chunk_result.visible[person_idx_in_chunk]
                )

            del imgs_tensor
            del padded_imgs
            del heatmaps
            del result_wrapper
            del keypoint_chunk_result

            gc.collect()
            # mm.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            on_progress(msg=f"Processed {i + 1}/{B} frames")

        if comfy_progress:
            pbar.update(1)

    if len(batch_pose_imgs) > 0:
        valid_len = len(batch_pose_imgs)
        process_frame_indices = batch_frame_indices

        with torch.no_grad():
            imgs_tensor = torch.stack(batch_pose_imgs, dim=0)
            padded_imgs = pad_tensor_to_batch_size(imgs_tensor, valid_len)
            heatmaps = batch_inference_topdown(
                sapiens, padded_imgs, dtype=torch.bfloat16
            )[:valid_len]

        result_wrapper = SapiensResult(
            heatmaps=heatmaps,
            centres=np.stack(batch_pose_img_centers, axis=0).astype(np.float32),
            scales=np.stack(batch_pose_img_scales, axis=0).astype(np.float32),
        )

        keypoint_chunk_result = _extract_keypoints(
            result_wrapper, use_torch=use_torch, udp_blur=udp_blur
        )

        for person_idx_in_chunk in range(valid_len):
            original_frame_idx = process_frame_indices[person_idx_in_chunk]
            acc = final_results_by_frame[original_frame_idx]
            acc["keypoints"].append(
                keypoint_chunk_result.keypoints[person_idx_in_chunk]
            )
            acc["scores"].append(keypoint_chunk_result.scores[person_idx_in_chunk])
            acc["visible"].append(keypoint_chunk_result.visible[person_idx_in_chunk])

        if comfy_progress:
            pbar.update(1)

    final_output = []

    for frame_idx in tqdm(range(B), desc="Assembling final results"):
        acc = final_results_by_frame[frame_idx]
        if not acc["keypoints"]:
            final_output.append(_create_empty_keypoint_result())
        else:
            if use_torch:
                final_kps = torch.stack(acc["keypoints"], dim=0).cpu().numpy()
                final_scores = torch.stack(acc["scores"], dim=0).cpu().numpy()
                final_visible = torch.stack(acc["visible"], dim=0).cpu().numpy()

                final_output.append(
                    KeypointResult(
                        keypoints=final_kps,
                        scores=final_scores,
                        visible=final_visible,
                    )
                )
            else:
                final_kps = np.stack(acc["keypoints"], axis=0)
                final_scores = np.stack(acc["scores"], axis=0)
                final_visible = np.stack(acc["visible"], axis=0)

                final_output.append(
                    KeypointResult(
                        keypoints=final_kps,
                        scores=final_scores,
                        visible=final_visible,
                    )
                )

    return add_metadata(final_output, (W, H))


def interpolate_keypoints(
    keypoints_a: torch.Tensor | np.ndarray,
    keypoints_b: torch.Tensor | np.ndarray,
    t: float | list[float] | torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """
    Perform linear interpolation between two sets of keypoints.

    This function is vectorized and works for single poses (308, 2), batches of
    poses (B, 308, 2), and can generate multiple interpolated frames at once.

    Args:
        keypoints_a: The starting keypoints. Shape (..., 308, 2).
        keypoints_b: The ending keypoints. Shape (..., 308, 2).
        t: The interpolation factor.
           - If a float, a single interpolated result is returned.
           - If a list or 1D tensor, multiple results are returned, one for
             each value in t. Values are clamped to the [0.0, 1.0] range.

    Returns:
        The interpolated keypoints.
        - If t is a float, shape is the same as inputs (..., 308, 2).
        - If t is a list/tensor of size N, shape is (N, ..., 308, 2).
    """
    if keypoints_a.shape != keypoints_b.shape:
        raise ValueError(
            f"Shapes of keypoints_a {keypoints_a.shape} and keypoints_b "
            f"{keypoints_b.shape} must match."
        )

    is_torch = isinstance(keypoints_a, torch.Tensor)

    if is_torch:
        device, dtype = keypoints_a.device, keypoints_a.dtype
        if not isinstance(keypoints_b, torch.Tensor):
            keypoints_b = torch.from_numpy(keypoints_b).to(device=device, dtype=dtype)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device, dtype=dtype)

        t = torch.clamp(t, 0.0, 1.0)
    else:
        if not isinstance(t, np.ndarray):
            t = np.array(t, dtype=keypoints_a.dtype)

        t = np.clip(t, 0.0, 1.0)

    if t.ndim == 0:
        return (1.0 - t) * keypoints_a + t * keypoints_b

    else:
        if is_torch:
            a = keypoints_a.unsqueeze(0)
            b = keypoints_b.unsqueeze(0)

            t_reshaped = t.view(-1, *([1] * a.ndim))
        else:
            a = np.expand_dims(keypoints_a, 0)
            b = np.expand_dims(keypoints_b, 0)
            t_reshaped = t.reshape(-1, *([1] * (a.ndim - 1)))

        return (1.0 - t_reshaped) * a + t_reshaped * b


def plot_animation_curves(
    results: list[KeypointResult],
    keypoint_names_to_plot: list[str] | list[int],
    person_id: int = 0,
    fps: int = 30,
    figsize: tuple[int, int] = (15, 8),
    title: str | None = None,
    all_keypoint_names: list[str] = GOLIATH_KEYPOINTS,
):
    """
    Plots the X and Y coordinate values of specified keypoints over time.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Please install matplotlib (`pip install matplotlib`) to use this function."
        )

    if not results or len(keypoint_names_to_plot) == 0:
        print("Warning: The results list is empty. Nothing to plot.")
        return

    use_id = isinstance(keypoint_names_to_plot[0], int)

    if use_id:
        indices_to_plot = keypoint_names_to_plot

    else:
        # --- 1. Prepare data for plotting ---
        keypoint_indices_map = {name: i for i, name in enumerate(all_keypoint_names)}

        indices_to_plot = [
            keypoint_indices_map[name]
            for name in keypoint_names_to_plot
            if name in keypoint_indices_map
        ]

        if not indices_to_plot:
            print(
                "Warning: None of the requested keypoint names were found. Nothing to plot."
            )
            return

    data_to_plot = {name: {"x": [], "y": []} for name in keypoint_names_to_plot}
    for frame_result in results:
        if person_id < frame_result.keypoints.shape[0]:
            kps = frame_result.keypoints[person_id]
            kps_np = kps.cpu().numpy() if hasattr(kps, "cpu") else np.asarray(kps)

            for name, index in zip(keypoint_names_to_plot, indices_to_plot):
                data_to_plot[name]["x"].append(kps_np[index, 0])
                data_to_plot[name]["y"].append(kps_np[index, 1])
        else:
            for name in keypoint_names_to_plot:
                data_to_plot[name]["x"].append(np.nan)
                data_to_plot[name]["y"].append(np.nan)
    fig, ax = plt.subplots(figsize=figsize)
    time_axis = np.arange(len(results)) / fps
    for name, coords in data_to_plot.items():
        ax.plot(time_axis, coords["x"], label=f"{name} X", linestyle="-")
        ax.plot(time_axis, coords["y"], label=f"{name} Y", linestyle="--")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Coordinate Value (pixels)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Animation Curves for Person {person_id}")

    plt.tight_layout()
    plt.show()


def smooth_keypoints(
    results: list[KeypointResult],
    window_length: int = 7,
    polyorder: int = 2,
) -> list[KeypointResult]:
    """
    Applies a Savitzky-Golay filter to smooth keypoint animations over time.

    This function should be called ONCE on the *entire* sequence of frames.
    """
    import numpy as np

    if window_length % 2 == 0 or window_length <= 0:
        raise ValueError("`window_length` must be a positive, odd integer.")
    if polyorder >= window_length:
        raise ValueError("`polyorder` must be less than `window_length`.")

    if len(results) < window_length:
        print(
            f"Warning: Data length ({len(results)}) is shorter than window_length "
            f"({window_length}). Filtering may be ineffective."
        )
        return results

    max_people = (
        max(r.keypoints.shape[0] for r in results if r.keypoints.ndim > 2)
        if results
        else 0
    )
    if max_people == 0:
        return results
    num_frames = len(results)
    num_keypoints = results[0].keypoints.shape[1]
    padded_keypoints = np.full(
        (num_frames, max_people, num_keypoints, 2), np.nan, dtype=np.float32
    )

    for i, frame_result in enumerate(results):
        num_people_in_frame = frame_result.keypoints.shape[0]
        if num_people_in_frame > 0:
            keypoints_np = (
                frame_result.keypoints.cpu().numpy()
                if hasattr(frame_result.keypoints, "cpu")
                else np.asarray(frame_result.keypoints)
            )
            padded_keypoints[i, :num_people_in_frame] = keypoints_np

    smoothed_padded_keypoints = np.copy(padded_keypoints)

    for p in range(max_people):
        for k in range(num_keypoints):
            valid_indices = np.isfinite(padded_keypoints[:, p, k, 0])
            if np.sum(valid_indices) >= window_length:
                smoothed_padded_keypoints[valid_indices, p, k, 0] = savgol_filter(
                    padded_keypoints[valid_indices, p, k, 0], window_length, polyorder
                )

            valid_indices = np.isfinite(padded_keypoints[:, p, k, 1])
            if np.sum(valid_indices) >= window_length:
                smoothed_padded_keypoints[valid_indices, p, k, 1] = savgol_filter(
                    padded_keypoints[valid_indices, p, k, 1], window_length, polyorder
                )
    smoothed_results = []
    for i in range(num_frames):
        original_num_people = results[i].keypoints.shape[0]
        if original_num_people == 0:
            smoothed_results.append(results[i])
            continue
        smoothed_kps = smoothed_padded_keypoints[i, :original_num_people]

        if isinstance(results[i].keypoints, torch.Tensor):
            smoothed_results.append(
                KeypointResult(
                    keypoints=torch.from_numpy(smoothed_kps).to(
                        device=results[i].keypoints.device
                    ),
                    scores=results[i].scores,
                    visible=results[i].visible,
                )
            )
        else:
            smoothed_results.append(
                KeypointResult(
                    keypoints=smoothed_kps,
                    scores=results[i].scores,
                    visible=results[i].visible,
                )
            )

    return smoothed_results


@lru_cache(maxsize=16)
def _get_savgol_coeffs(
    window_length: int, polyorder: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Calculates and caches Sav-Gol filter coefficients.
    This is called only once per unique set of parameters.
    """
    try:
        from scipy.signal import savgol_coeffs
    except ImportError:
        raise ImportError(
            "Please install SciPy (`pip install scipy`) to use the Savitzky-Golay filter."
        )

    coeffs = savgol_coeffs(window_length, polyorder)
    return torch.from_numpy(coeffs).to(device=device, dtype=dtype).view(1, 1, -1)


def _extract_keypoints(
    heatmaps: SapiensResult,
    *,
    pbar=None,
    on_progress: ProgressCallable | None = None,
    use_torch=False,
    udp_blur=11,
) -> KeypointResult:
    if on_progress is None:
        on_progress = progress_fallback

    if pbar is not None:
        _start = pbar.current

    def _on_update(i):
        if pbar is not None:
            pbar.update(i)
            on_progress(
                msg=f"Frame {pbar.current - _start}/{pbar.total - _start}",
                title="Extract Keypoints",
            )

    with timed("extract keypoints"):
        if use_torch:
            from .sapiens_utils import extract_keypoints_torch as extract_keypoints
        else:
            from .sapiens_utils import extract_keypoints

        keypoints = extract_keypoints(
            heatmaps, udp_blur=udp_blur
        )  # , on_update=_on_update)

        return keypoints


# @jaxtyped(typechecker=typechecker)
def draw(
    result: Result,
    bboxes_per_frame: list[BBox] | None = None,
    image: torch.Tensor | None = None,
    output_size: tuple[int, int] | None = None,
    draw_on_image=False,
    keypoints_radius=3,
    bone_thickness=3,
    keypoints_threshold=0.3,
    palettes=None,
    style: Literal["sapiens", "openpose"] = "sapiens",
    verbose=False,
    draw_face: Literal["off", "clean", "full"] = "full",
    draw_clavicle=False,
    draw_upper_body=True,
    draw_lower_body=True,
    draw_labels=None,
    label_size=0.5,
    draw_keypoints=True,
    draw_skeleton=True,
    draw_bbox=False,
    auto_scale=True,
    bitmask: torch.Tensor | None = None,
    # return_debug_data=False,
) -> UInt8[LazyProxyTensor, "B H W C"]:
    """Draw all people on all frames.

    thickness and radius is normalized by the tensor H/W dimensions
    """

    from .classes_and_palettes import (
        GOLIATH_KPTS_COLORS,
        GOLIATH_SKELETON_INFO,
    )
    from .utils import tensor_info, draw_images
    import numpy as np

    num_frames = len(result.data)

    if image is not None:
        h, w = image.shape[1], image.shape[2]
    elif output_size is not None:
        h, w = output_size

    else:
        w, h = result.metadata["input_dimensions"]

    images_out_tensor = torch.empty(
        (num_frames, h, w, 3), dtype=torch.uint8, device="cpu"
    )
    if verbose:
        print(f"Pre-allocated output tensor with shape: {images_out_tensor.shape}")

    if verbose:
        print(tensor_info(image, "RGB Frames"))

    if palettes is not None:
        KPTS_COLORS, SKELETON_INFO = palettes

    else:
        KPTS_COLORS = GOLIATH_KPTS_COLORS
        SKELETON_INFO = GOLIATH_SKELETON_INFO

    chunk_idx = 0
    person_idx_in_chunk = 0

    if auto_scale:
        keypoints_radius = calculate_proportional_radius((h, w), keypoints_radius)
        bone_thickness = calculate_proportional_radius((h, w), bone_thickness)

    if bboxes_per_frame is None:
        _bboxes_per_frame = [None] * num_frames
    else:
        _bboxes_per_frame = bboxes_per_frame

    data = result.data

    source_w, source_h = result.metadata["input_dimensions"]
    if source_w != w or source_h != h:
        scale_w = w / source_w
        scale_h = h / source_h

        scale = min(scale_w, scale_h)

        scaled_w = source_w * scale
        scaled_h = source_h * scale

        offset_x = (w - scaled_w) / 2
        offset_y = (h - scaled_h) / 2

        remapped_data = []
        for kpt_res in result.data:
            original_keypoints = kpt_res.keypoints

            if isinstance(original_keypoints, torch.Tensor):
                offset = torch.tensor(
                    [offset_x, offset_y],
                    dtype=original_keypoints.dtype,
                    device=original_keypoints.device,
                )
                remapped_keypoints = original_keypoints * scale + offset
            else:
                offset = np.array([offset_x, offset_y], dtype=original_keypoints.dtype)
                remapped_keypoints = original_keypoints * scale + offset

            remapped_data.append(
                KeypointResult(
                    keypoints=remapped_keypoints,
                    scores=kpt_res.scores,
                    visible=kpt_res.visible,
                )
            )

        data = remapped_data

    for frame_idx, kpt_res in enumerate(data):

        num_people_in_frame = kpt_res.keypoints.shape[0]  # [frame_idx]
        base_canvas = image[frame_idx] if draw_on_image and image is not None else None
        bboxes = _bboxes_per_frame[frame_idx]

        people_to_draw = []
        for _ in range(num_people_in_frame):
            if chunk_idx >= len(data) or person_idx_in_chunk >= len(
                data[chunk_idx].keypoints
            ):
                chunk_idx += 1
                person_idx_in_chunk = 0
            current_chunk = data[chunk_idx]
            people_to_draw.append(
                KeypointResult(
                    keypoints=current_chunk.keypoints[person_idx_in_chunk][None, ...],
                    scores=current_chunk.scores[person_idx_in_chunk][None, ...],
                    visible=current_chunk.visible[person_idx_in_chunk][None, ...],
                )
            )
            person_idx_in_chunk += 1

        final_drawn_frame = base_canvas
        for person_kpts in people_to_draw:
            # if return_debug_data:
            #     return (person_kpts, bboxes, final_drawn_frame, (w, h))

            final_drawn_frame = draw_images(
                person_kpts,
                img=final_drawn_frame,
                target_size=(w, h),
                draw_on_black=not draw_on_image,
                kpt_colors=KPTS_COLORS,
                kpt_thr=keypoints_threshold,
                radius=keypoints_radius,
                skeleton_info=SKELETON_INFO,
                thickness=bone_thickness,
                style=style,
                draw_face=draw_face,
                draw_clavicle=draw_clavicle,
                draw_upper_body=draw_upper_body,
                draw_lower_body=draw_lower_body,
                draw_labels=draw_labels,
                label_size=label_size,
                bitmask=bitmask,
                draw_keypoints=draw_keypoints,
                draw_skeleton=draw_skeleton,
                draw_bbox=draw_bbox,
                bboxes=bboxes,
            )

        if final_drawn_frame is None:
            final_drawn_frame = np.zeros((h, w, 3), dtype=np.uint8)
        elif isinstance(final_drawn_frame, torch.Tensor):
            final_drawn_frame = (final_drawn_frame.cpu().numpy() * 255).astype(np.uint8)

        images_out_tensor[frame_idx] = torch.from_numpy(final_drawn_frame)

    return LazyProxyTensor(images_out_tensor)  # .float() / 255.0
