from contextlib import contextmanager
import textwrap
from typing import Any
from types import EllipsisType
import numpy as np
import warnings
import torch


class LazyProxyTensor:
    """Memory-efficient proxy that wrap a tensor but presents itself as a different dtype (e.g., float32).

    It mimics a torch.Tensor's read-only attributes and methods. Data conversion
    and normalization happen lazily on access (e.g., via slicing), avoiding
    the high memory cost of a full conversion.

    Supported source dtypes:
    - torch.uint8 (normalized from [0, 255])
    - torch.uint16 (normalized from [0, 65535])
    - All float types (passed through, assumed to be in [0, 1] range)"
    """

    _source_tensor: torch.Tensor
    _target_dtype: torch.dtype
    _target_element_size: int
    _scale_divisor: float
    _warned_inefficient_access: bool

    def __init__(self, source_tensor, target_dtype=torch.float32, target_device=None):
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor.")

        self._source_tensor = source_tensor
        self._target_dtype = target_dtype
        self._target_device = (
            target_device if target_device is not None else source_tensor.device
        )

        # Determine the normalization divisor based on source dtype
        # fmt: off
        if source_tensor.dtype == torch.uint8: self._scale_divisor = 255.0
        elif source_tensor.dtype == torch.uint16: self._scale_divisor = 65535.0
        elif torch.is_floating_point(source_tensor): self._scale_divisor = 1.0
        else: raise ValueError(f"Unsupported source dtype for LazyProxyTensor: {source_tensor.dtype}")
        # fmt: on

        self._target_element_size = torch.empty(
            (), dtype=self._target_dtype
        ).element_size()
        self._warned_inefficient_access = False

    def is_contiguous(self, *args, **kwargs):
        return self._source_tensor.is_contiguous(*args, **kwargs)

    def stride(self, *args, **kwargs):
        return self._source_tensor.stride(*args, **kwargs)

    @property
    def shape(self):
        return self._source_tensor.shape

    @property
    def size(self):
        return self._source_tensor.size

    def to(self, target):
        if isinstance(target, torch.dtype):
            self._target_dtype = target
            return self

        transfered = self._source_tensor.to(target)
        return LazyProxyTensor(transfered, self._target_dtype)

    @property
    def requires_grad(self):
        return False

    def nelement(self):
        """Return the total number of elements in the (pretend) tensor."""
        return self._source_tensor.nelement()

    def element_size(self):
        """Return the size in bytes of an individual (pretend) float element."""
        return self._target_element_size

    @property
    def dtype(self):
        return self._target_dtype

    @property
    def device(self):
        return self._target_device

    def __len__(self):
        return self._source_tensor.shape[0]

    def __setitem__(self, key, value):
        """
        Lazily WRITES a slice, converting the value back to the source format.
        This method is what makes `proxy[key] = value` work.
        """
        # Ensure the incoming value is a tensor on the correct target device
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(
                value, dtype=self._target_dtype, device=self._target_device
            )
        elif value.device != self._target_device:
            value = value.to(self._target_device)

        # --- The Streamlined "Reverse Conversion" Pipeline ---
        converted_value = value * self._scale_divisor
        source_dtype = self._source_tensor.dtype

        if torch.is_floating_point(self._source_tensor):
            # If the source is already a float, just cast it.
            converted_value = converted_value.to(source_dtype)
        else:
            # For integer types, use torch.iinfo to get type limits generically
            # This handles uint8, uint16, etc. automatically.
            type_info = torch.iinfo(source_dtype)
            converted_value = (
                converted_value.round()
                .clamp(type_info.min, type_info.max)
                .to(source_dtype)
            )

        # 3. Move the correctly formatted value back to the source device for assignment
        converted_value = converted_value.to(self._source_tensor.device)

        # 4. Perform the actual assignment on the underlying tensor
        self._source_tensor[key] = converted_value

    @contextmanager
    def edit(self):
        real_tensor = self[:]

        try:
            yield real_tensor
        finally:
            self[:] = real_tensor

    def __getitem__(self, key):
        if (
            self._source_tensor.device != self._target_device
            and not self._warned_inefficient_access
        ):
            warnings.warn(
                "Inefficient access pattern detected for LazyProxyTensor. "
                "You are slicing a device-proxied tensor, which causes slow, "
                "repeated data transfers. For performance, use the .iter_chunks() method."
            )
            self._warned_inefficient_access = True

        subset = self._source_tensor[key]

        return (
            subset.to(self._target_device).to(self._target_dtype) / self._scale_divisor
        )

    def iter_chunks(self, chunk_size=16):
        for i in range(0, len(self), chunk_size):
            chunk = self._source_tensor[i : i + chunk_size]
            yield (
                chunk.to(self._target_device, non_blocking=True).to(self._target_dtype)
                / self._scale_divisor
            )

    def squeeze(self, dim: str | EllipsisType | None = None):
        squeezed = self._source_tensor.squeeze(dim)
        return LazyProxyTensor(squeezed, self._target_dtype)

    def unsqueeze(self, dim: int = 0):
        unsqueezed = self._source_tensor.unsqueeze(dim)
        return LazyProxyTensor(unsqueezed, self._target_dtype)

    def repeat(self, *sizes):
        repeated = self._source_tensor.repeat(*sizes)
        return LazyProxyTensor(repeated, self._target_dtype)

    def permute(self, *dims):
        permuted = self._source_tensor.permute(*dims)
        return LazyProxyTensor(permuted, self._target_dtype)

    def _format_mem_size(self, mem_bytes):
        if mem_bytes > 1e9:
            return f"{mem_bytes / 1e9:.2f} GB"
        if mem_bytes > 1e6:
            return f"{mem_bytes / 1e6:.2f} MB"
        if mem_bytes > 1e3:
            return f"{mem_bytes / 1e3:.2f} KB"
        return f"{mem_bytes} B"

    def __repr__(self):
        # actual_mem_bytes = (
        #     self._source_tensor.element_size() * self._source_tensor.nelement()
        # )
        #
        # potential_mem_bytes = self.element_size() * self.nelement()
        #
        # actual_dtype_str = str(self._source_tensor.dtype).replace("torch.", "")
        # target_dtype_str = str(self.dtype).replace("torch.", "")

        actual_info = tensor_info(self._source_tensor, name="Source")
        target_info = tensor_info(self, name="Target")

        info = f"""
        {target_info}

        {actual_info}
        """
        return textwrap.dedent(info).strip()


def tensor_info(
    t: torch.Tensor | LazyProxyTensor | np.ndarray | None, name=None, *, mem=True
):
    if t is None:
        return

    name = name or "Unnamed"
    is_tensor = isinstance(t, torch.Tensor | LazyProxyTensor)
    mem_str = "N/A"
    if mem:
        if is_tensor:
            mem_bytes = t.element_size() * t.nelement()
        else:
            mem_bytes = t.itemsize * t.size

        if mem_bytes > 1e9:
            mem_str = f"{mem_bytes / 1e9:.2f} GB"
        elif mem_bytes > 1e6:
            mem_str = f"{mem_bytes / 1e6:.2f} MB"
        elif mem_bytes > 1e3:
            mem_str = f"{mem_bytes / 1e3:.2f} KB"
        else:
            mem_str = f"{mem_bytes} B"

    device = "N/A"
    grad = "False"
    type_name = "Tensor" if is_tensor else "Numpy Array"

    if isinstance(t, torch.Tensor):
        device = t.device
        grad = str(t.requires_grad)

    return f"""{type_name} {name}
    shape: {t.shape}
    dtype: {str(t.dtype).replace("torch.", "")}
    device: {device}
    requires grad: {grad}
    memory: {mem_str}
    """


def is_tensor(t: Any) -> bool:
    return isinstance(t, torch.Tensor | LazyProxyTensor)
