# ruff: noqa: F722 - Syntax error in forward annotation: {parse_error}
"""WIP v2 of the pipeline, fully batching everything and not storing the cropped images"""

from dataclasses import dataclass
from typing import Callable

# from memory_profiler import profile

from .datatypes import BFloat16, Bool
import torchvision.transforms.functional as TF

from .constants import PREPROCESS_H, PREPROCESS_W

from .utils import chunked_tensors, print_ram, tensor_info
from .datatypes import SapiensResult
from torch.nn.utils.rnn import pad_sequence

import torch


# @jaxtyped(typechecker=typechecker)
def get_heatmap_maximum_5d(
    padded_heatmaps: BFloat16[torch.Tensor, "B N 308 256 192"],
    validity_mask: Bool[torch.Tensor, "B N"],  # | None = None,
    *,
    offload=None,
    verbose=False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Get maximum response location and value from a 5D batch of heatmaps.

    Args:
        padded_heatmaps (torch.Tensor): Padded heatmaps tensor of shape
            (B, N_max, K, H, W).
        validity_mask (torch.Tensor): Boolean tensor of shape (B, N_max)
            where `True` indicates a real person and `False` indicates padding.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - locs (torch.Tensor): Locations of maximums for the entire batch,
            shape (B, N_max, K, 2) in (x, y) format. Padded/invalid entries are [-1, -1].
        - vals (torch.Tensor): Values of maximums for the entire batch,
            shape (B, N_max, K). Padded/invalid entries are 0.0.
    """
    if verbose:
        print(tensor_info(padded_heatmaps, "Padded Heatmaps"))
        if validity_mask is not None:
            print(tensor_info(validity_mask, "Validity Mask"))

    B, N_max, K, H, W = padded_heatmaps.shape
    assert validity_mask.shape == (B, N_max), f"Mask shape must be ({B}, {N_max})"

    # mask reshaped to (B, N_max, 1, 1, 1) so it broadcasts correctly
    # over the (B, N_max, K, H, W) heatmaps tensor.
    masked_heatmaps = torch.where(
        validity_mask.view(B, N_max, 1, 1, 1), padded_heatmaps, -torch.inf
    )

    heatmaps_flat = masked_heatmaps.view(B, N_max, K, -1)
    vals, indices = torch.max(heatmaps_flat, dim=3)

    # flatten
    x_locs = indices % W
    y_locs = indices // W
    locs = torch.stack([x_locs, y_locs], dim=3).to(torch.float32)

    # filter both padded frames and low score ones
    invalid_loc_mask = vals <= 0.0
    locs[invalid_loc_mask] = -1.0

    vals = torch.where(validity_mask.unsqueeze(-1), vals, 0.0)

    if verbose:
        print("Keypoints[0]:")
        print(locs[0][0][0])

        print("Scores[0]:")
        print(vals[0][0][0])

    if offload:
        return locs.to(offload), vals.to(offload), validity_mask.to(offload)

    return locs, vals, validity_mask


def refine(keypoints, heatmaps, blur_kernel_size=11):
    # N, K = keypoints.shape[:2]
    _F, _B, N, K = keypoints.shape
    _F, _B, _K, H, W = heatmaps.shape  # [1:]

    print("Computing blur")
    heatmaps = torch.stack(
        [TF.gaussian_blur(x, [blur_kernel_size, blur_kernel_size]) for x in heatmaps],
        dim=0,
    )

    print("Padding")
    torch.clip(heatmaps, 1e-3, 50.0, out=heatmaps)
    torch.log(heatmaps, out=heatmaps)

    # heatmaps_pad = TF.pad(
    #     heatmaps, [0, 0, 1, 1, 1, 1], padding_mode="edge"
    # ).flatten()
    heatmaps_pad = heatmaps.flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * torch.arange(0, K, device=index.device)
        index = index.int().reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = torch.concatenate([dx, dy], dim=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = torch.concatenate([dxx, dxy, dxy, dyy], dim=1)
        hessian = hessian.reshape(K, 2, 2)

        identity = torch.eye(2, device="cuda", dtype=torch.float32)
        hessian_inv = torch.linalg.inv(
            hessian + torch.finfo(torch.float32).eps * identity
        )

        update_step = hessian_inv @ derivative.unsqueeze(-1)

        keypoints[n] -= update_step.squeeze(-1)

    return keypoints


# @jaxtyped(typechecker=typechecker)


# @profile
# @jaxtyped(typechecker=typechecker)
def extract_keypoints(
    results: list[SapiensResult],
    *,
    input_shape: tuple[int, int] = (PREPROCESS_W, PREPROCESS_H),
    heatmap_scale: int = 4,
    # verbose=False
    on_update: Callable[[int], None] | None = None,
):
    _heatmaps = [x.heatmaps for x in results]
    _centres = [torch.from_numpy(x.centres) for x in results]
    _scales = [torch.from_numpy(x.scales) for x in results]

    _num_people_per_frame = [h.shape[0] for h in _heatmaps]

    # _padded_heatmaps = pad_sequence(_heatmaps, batch_first=True, padding_value=0.0)

    # max_len = max([x.shape[0] for x in _heatmaps])
    # C, H, W = _heatmaps[0].shape[1:]
    # _padded_heatmaps = torch.zeros(
    #     (len(_heatmaps), max_len, C, H, W), dtype=_heatmaps[0].dtype
    # )
    # for i, h in enumerate(_heatmaps):
    #     _padded_heatmaps[i, : h.shape[0]] = h
    # _padded_heatmaps = torch.stack(_heatmaps, dim=0)
    # _padded_centres = pad_sequence(_centres, batch_first=True, padding_value=0.0)

    # _padded_scales = pad_sequence(_scales, batch_first=True, padding_value=0.0)

    # print(tensor_info(_padded_heatmaps))

    # del _centres
    # del _scales
    # del _heatmaps
    # gc.collect()

    _B = len(_num_people_per_frame)
    _N_max = max(_num_people_per_frame)
    # _N_max = _padded_heatmaps.shape[1]

    _range_tensor = torch.arange(_N_max)

    # Create a tensor of the original lengths, and reshape for broadcasting
    _lengths_tensor = torch.tensor(_num_people_per_frame).unsqueeze(1)

    # Use broadcasting to compare the range with the lengths
    # This creates a mask of shape [B, N_max]
    _mask = _range_tensor < _lengths_tensor

    _locs = []
    _final_vals = []

    @dataclass
    class Data:
        heatmaps: torch.Tensor
        centres: torch.Tensor
        scales: torch.Tensor
        mask: torch.Tensor

    @dataclass
    class Result:
        locs: torch.Tensor
        final_vals: torch.Tensor
        # scales: torch.Tensor
        # mask: torch.Tensor

    _data = []

    W, H = (
        int(input_shape[0] / heatmap_scale),
        int(input_shape[1] / heatmap_scale),
    )

    print_ram("before chunking")
    with torch.no_grad():
        for t, length in chunked_tensors(
            Data,
            heatmaps=_heatmaps,
            scales=_scales,
            centres=_centres,
            mask=_mask,
        ):
            # decode the heatmaps
            tensor_info(t.heatmaps, "HEATMAPS CHUNK")
            tensor_info(t.mask, "MASK")
            keypoints, scores, visible = get_heatmap_maximum_5d(
                pad_sequence(
                    t.heatmaps[:length].cuda(), batch_first=True, padding_value=0.0
                ),
                t.mask[:length].cuda(),
            )

            # TODO: refine in torch
            # keypoints = refine(keypoints, t.heatmaps[:length].cuda(), blur_kernel_size=11)

            device = keypoints.device
            dtype = keypoints.dtype

            print(tensor_info(keypoints, "KEYPOINTS CHUNK"))
            print(tensor_info(t.centres, "PADDED CENTRES CHUNK"))
            print(tensor_info(t.scales, "PADDED SCALES CHUNK"))

            keypoints[..., 0] = (keypoints[..., 0] / (W - 1)) * input_shape[0]
            keypoints[..., 1] = (keypoints[..., 1] / (H - 1)) * input_shape[1]

            # scale and center keypoints
            scales_bc = t.scales[:length].unsqueeze(2).to(device)
            centres_bc = t.centres[:length].unsqueeze(2).to(device)
            input_shape_tensor = torch.tensor(input_shape, device=device, dtype=dtype)

            keypoints = (
                (keypoints / input_shape_tensor) * scales_bc
                + centres_bc
                - 0.5 * scales_bc
            )

            _data.append((keypoints.cpu(), scores.float().cpu(), visible.cpu()))

            if on_update is not None:
                on_update(length)

        # del _padded_scales
        # del _padded_heatmaps
        # del _padded_centres

        return _data
