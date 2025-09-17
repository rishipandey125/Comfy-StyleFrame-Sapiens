from pathlib import Path
from typing import List, Any

import torch

from .classes_and_palettes import GOLIATH_KEYPOINTS
from .datatypes import KeypointResult, Result
from .utils import log
import numpy as np
import json

VERSION = "1.0.2"
HAS_MSG_PACK = True

try:
    import msgpack
    import msgpack_numpy as m

    # NOTE: needed for automatic serialization/deserialization of numpy arrays
    m.patch()
except ImportError:
    HAS_MSG_PACK = False
    msgpack = None
    log.info(
        "msgpack or msgpack_numpy not found. .msgpack format will not be available."
    )


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        # fmt: off
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        #fmt: on
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _ensure_msgpack():
    if not msgpack:
        raise RuntimeError("Please install msgpack to load .msgpack files")


def _ensure_extension(path: Path):
    if path.suffix not in [".json", ".msgpack"]:
        raise RuntimeError(
            f"Unsupported file extension: {path.suffix}, please use .json or .msgpack"
        )


def load_animation(path: Path) -> Result:
    _ensure_extension(path)

    if path.suffix == ".json":
        with open(path, "r") as f:
            unpacked_data = json.load(f)
    elif path.suffix == ".msgpack":
        _ensure_msgpack()
        with open(path, "rb") as f:
            unpacked_data = msgpack.unpackb(f.read(), raw=False)

    log.info(f"Successfully loaded raw data from {path}")
    return _deserialize_animation(unpacked_data)


def save_animation(
    output_path: Path,
    result: Result,
):
    metadata, keypoints = result
    _ensure_extension(output_path)
    packed_data = _serialize_animation(keypoints, metadata)

    if output_path.suffix == ".json":
        with output_path.open("w") as f:
            json.dump(packed_data, f, cls=NumpyEncoder)  # , indent=2)
    elif output_path.suffix == ".msgpack":
        _ensure_msgpack()
        with output_path.open("wb") as f:
            # Use msgpack.packb to serialize the data
            packed_data = msgpack.packb(packed_data, use_bin_type=True)
            if not packed_data:
                raise ValueError("Packed data is null")
            f.write(packed_data)

    log.info(f"Animation data successfully saved to {output_path}")


def _deserialize_animation(
    # input_path: Path,
    unpacked_data: dict[str, Any],
    remap_dimension: tuple[int, int] | None = None,
) -> Result:
    # with input_path.open("rb") as f:
    #     packed_data = f.read()
    #     # Use msgpack.unpackb to deserialize the data
    #     unpacked_data = msgpack.unpackb(packed_data, raw=False)

    metadata = unpacked_data["metadata"]
    frames = unpacked_data["frames"]

    animation_result: list[KeypointResult] = []
    for frame_data in frames:
        num_people = len(frame_data["people"])
        if num_people == 0:
            continue

        keypoints_batch = np.array(
            [p["keypoints"] for p in frame_data["people"]], dtype=np.float32
        )
        scores_batch = np.array(
            [p["scores"] for p in frame_data["people"]], dtype=np.float32
        )
        visible_batch = np.array(
            [p["visible"] for p in frame_data["people"]], dtype=np.float32
        )

        animation_result.append(
            KeypointResult(
                keypoints=torch.from_numpy(keypoints_batch),
                scores=scores_batch,
                visible=visible_batch,
            )
        )

    if remap_dimension is not None:
        animation_result = _normalize_keypoints(
            animation_result, remap_dimension, invert=True
        )

    if metadata["version"] == "1.0.1":
        # adding input dim
        metadata["input_dimensions"] = (1080, 1920)

    log.info(f"Successfully deserialized {len(animation_result)} frames.")
    return Result(metadata, animation_result)


def add_metadata(
    animation_data: List[KeypointResult], input_dimensions: tuple[int, int], **extra
) -> Result:
    full_metadata = {
        "version": VERSION,
        "total_frames": len(animation_data),
        "keypoint_names": GOLIATH_KEYPOINTS,
        "input_dimensions": input_dimensions,
        **extra,
    }

    return Result(full_metadata, animation_data)


def _serialize_animation(
    animation_data: List[KeypointResult],
    metadata: dict[str, Any],
    remap_dimension: tuple[int, int] | None = None,
):
    """
    Serializes a list of KeypointResult objects to a MessagePack file.

    Args:
        animation_data: A list of KeypointResult objects for each frame.
        output_path: The path to save the output .msgpack file.
        metadata: Optional dictionary for additional metadata.
        remap_dimension: If provided this will normalize the keypoints before saving them
    """


    frames_payload = []
    for i, frame_result in enumerate(animation_data):
        keypoints_np = (
            frame_result.keypoints.cpu().numpy()
            if isinstance(frame_result.keypoints, torch.Tensor)
            else np.asarray(frame_result.keypoints)
        )
        scores_np = (
            frame_result.scores.cpu().numpy()
            if isinstance(frame_result.scores, torch.Tensor)
            else np.asarray(frame_result.scores)
        )
        visible_np = (
            frame_result.visible.cpu().numpy()
            if isinstance(frame_result.visible, torch.Tensor)
            else np.asarray(frame_result.visible)
        )

        people_payload = []
        num_people = keypoints_np.shape[0]

        for j in range(num_people):
            person_data = {
                "person_id": j,
                "keypoints": keypoints_np[j],
                "scores": scores_np[j],
                "visible": visible_np[j],
            }
            people_payload.append(person_data)

        frames_payload.append({"frame_number": i, "people": people_payload})

    return {"metadata": metadata, "frames": frames_payload}


