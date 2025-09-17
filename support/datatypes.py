# ruff: noqa: F722 - Syntax error in forward annotation: jaxtyping
# ruff: noqa: F401 - `{name}` imported but unused; consider using `importlib.util.find_spec` to test for availability

from dataclasses import dataclass, asdict, fields, is_dataclass
from typing import (
    Annotated,
    Any,
    NamedTuple,
    Protocol,
    TypeVar,
    TypedDict,
    runtime_checkable,
)

try:
    from jaxtyping import (
        # for order
        Bool,
        Float,
        BFloat16,
        Float32,
        UInt8,
        Int,
        jaxtyped,
    )
    from beartype import beartype as typechecker

except Exception:
    from typing import (
        Annotated as Bool,
        Annotated as Float,
        Annotated as BFloat16,
        Annotated as Float32,
        Annotated as UInt8,
        Annotated as Int,
    )


import torch
import numpy as np


# from .yolo import Detector
from .proxy_tensor import LazyProxyTensor


@runtime_checkable
class ProgressCallable(Protocol):
    def __call__(self, *, it=None, msg=None, title=None) -> None: ...


class LinkDict(dict):
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if step is None:
                step = 1
            for i in range(start, stop, step):
                if i in self:
                    self[i]["color"] = value
        else:
            super().__setitem__(key, value)


OneLink = int | tuple[int, int]
MidLink = tuple[OneLink, OneLink]
Link = tuple[OneLink | tuple[OneLink, OneLink], OneLink | tuple[OneLink, OneLink]]


class LinkInfo(TypedDict):
    # connected points
    link: Link
    id: int
    color: tuple[int, int, int]


class SerializableMixin:
    """
    A mixin for dataclasses to make them serializable to dicts
    and iterable for unpacking.
    """

    def __getstate__(self):
        if not is_dataclass(self):
            raise TypeError("SerializableMixin can only be used with dataclasses.")
        return asdict(self)

    def __iter__(self):
        if not is_dataclass(self):
            raise TypeError("SerializableMixin can only be used with dataclasses.")
        for f in fields(self):
            yield getattr(self, f.name)


@dataclass(frozen=True)
class SapiensResult(SerializableMixin):
    """
    Holds the output of the Sapiens model.

    Attributes:
        heatmaps: Predicted heatmaps for keypoints.
        centres: Predicted bbox centres.
        scales: Predicted bbox scales.
    """

    heatmaps: BFloat16[torch.Tensor, "?N 308 256 192"]
    centres: Float32[np.ndarray, "?N 2"]
    scales: Float32[np.ndarray, "?N 2"]


@dataclass(frozen=True)
class KeypointResult(SerializableMixin):
    """
    Holds final keypoint detection results.

    Attributes:
        keypoints: Final (x, y) coordinates of keypoints.
        scores: Confidence scores for each keypoint.
        visible: Visibility flags for each keypoint.
    """

    keypoints: Float32[np.ndarray | torch.Tensor, "B 308 2"]
    scores: Float32[np.ndarray | torch.Tensor, "B 308"]
    visible: Float32[np.ndarray | torch.Tensor, "B 308"]


@dataclass(frozen=True)
class Result(SerializableMixin):
    metadata: dict[str, Any]
    data: list[KeypointResult]


BBox = Float32[np.ndarray | torch.Tensor, "N 4"]

BBoxPerFrame = Annotated[list[BBox], "List of Bboxes (N,4) per frame"]


Models = tuple[
    Annotated[torch.nn.Module, "Sapiens"],
    Annotated[torch.nn.Module | Any, "BBox Detector"],
]

AnyTensor = np.ndarray | torch.Tensor | LazyProxyTensor
# A single tensor
ImageTensor = UInt8[AnyTensor, "H W C"]
ImageTensorBatch = UInt8[torch.Tensor | LazyProxyTensor, "B H W C"]

ImageTensorBatchChannelFirst = UInt8[torch.Tensor | LazyProxyTensor, "B C H W"]


# WARN: these types aren't matching yet between mmdet and yolo

DType = TypeVar("DType", bound=np.generic, covariant=True)
KeypointDict = dict[str, tuple[float, float, float]]

KeypointsPerPerson = KeypointDict
KeypointsPerFrame = KeypointDict | list[KeypointDict]
KeypointsBatch = list[KeypointsPerFrame]

# BboxPerPerson = Bbox
# BboxesPerFrame = Bbox | list[Bbox]
# BboxesBatch = list[BboxesPerFrame]


# List of bboxes for a single frame
# FrameBboxes = list[Bbox]
# List of frames, each with list of bboxes
# BatchBboxes = list[FrameBboxes]
# List of keypoints for a single frame
# FrameKeypoints = list[KeypointDict]
# List of frames, each with list of keypoints
# BatchKeypoints = list[FrameKeypoints]
