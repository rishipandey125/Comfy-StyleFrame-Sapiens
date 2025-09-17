# ruff: noqa: F722 - Syntax error in forward annotation: jaxtyping
# from memory_profiler import profile

# region imports
import json
from pathlib import Path
from typing import Literal
import comfy.utils
import torch

from .support.datatypes import Result

from .support.io import add_metadata


from .support.classes_and_palettes import GOLIATH_KEYPOINTS
from .support.utils import setup_models, log, timed_comfy_node
import numpy as np

try:
    from server import PromptServer
except ModuleNotFoundError:
    PromptServer = None
# endregion

# Add sapiens to folder_path
setup_models()


# TODO: improve the look of this
class SapiensBase:
    unique_id = None
    _msg = ""

    def send_progress(self, msg, *, add=False, title: str | None = None):
        if not add:
            self._msg = ""

        if title is not None:
            self._msg += f'<b style=""font-size: 1.2em;">{title}</b>'

        self._msg += f"<p>{msg}</p>"

        if self.unique_id and PromptServer:
            PromptServer.instance.send_progress_text(self._msg, self.unique_id)

        log.info(self._msg)


# region main nodes


class SAPIENS_LoadPoseModel(SapiensBase):
    """
    ComfyUI Node to load a pose estimation model.
    Supports TorchScript and torch.export models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_size": (
                    [
                        # NOTE: not good enough?
                        # "0.3B",
                        # "0.6B",
                        "1B"
                    ],
                    {"default": "1B"},
                ),
                "use_torchscript": ("BOOLEAN", {"default": False}),
                "compile": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "slower on first run bust faster on subsequent ones",
                    },
                ),
                "bbox_backend": (
                    ("auto", "RtmDet", "Yolov8"),
                    {
                        "default": "auto",
                        "tooltip": "Auto will use RtmDet if the libraries are installed else fallback to Yolov8",
                    },
                ),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("SAPIENS_MODEL",)
    RETURN_NAMES = ("pose model",)
    FUNCTION = "load_model"
    CATEGORY = "SapiensPose"

    @timed_comfy_node("Load Pose Model")
    def load_model(
        self,
        *,
        model_size="1B",
        compile=True,
        use_torchscript=False,
        device="cuda",
        unique_id=None,
        bbox_backend="auto",
        auto_download=False,
    ):
        from .support.pipeline import load_model

        self.unique_id = unique_id

        def comfy_progress(it=None, msg=None, title=None):
            if msg:
                self.send_progress(msg, title=title)

        model = load_model(
            model_size,
            compile=compile,
            use_torchscript=use_torchscript,
            device=device,
            auto_download=auto_download,
            bbox_backend=bbox_backend,
            on_progress=comfy_progress,
            comfy_progress=True,
        )

        return (model,)


class SAPIENS_Preprocessor(SapiensBase):
    """
    Preprocess frames for pose estimation (bbox detection)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS_MODEL",),
                "image": ("IMAGE",),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "step": 0.01,
                        "tooltip": "Confidence threshold to detect bounding boxes",
                    },
                ),
                "nms_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "step": 0.01,
                        "tooltip": "NMS threshold to detect and merge overlapping bboxes",
                    },
                ),
            },
            "optional": {
                "bbox_padding": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "step": 0.01,
                        "tooltip": "Padding to add to the bbox",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # TODO: add ExtractCrop
    RETURN_TYPES = ("SAPIENS_BBOXES",)
    RETURN_NAMES = ("bboxes",)
    # OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = (
        # "Preview the crops, the order not being guaranteed if more than 1 character this is mostly useful to debug the crops",
        "This contains the bbox data (crop, centers, scales and frame mapping)",
    )
    FUNCTION = "preprocess"
    CATEGORY = "SapiensPose"

    @timed_comfy_node("Sapiens Preprocessor")
    def preprocess(
        self,
        *,
        model,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.3,
        bbox_padding: float = 0.1,
        image: torch.Tensor,
        unique_id=None,
    ):
        from .support.pipeline import preprocess

        self.unique_id = unique_id
        self._msg = ""

        pbar = comfy.utils.ProgressBar(image.shape[0])

        def comfy_progress(it=None, msg=None, title=None):
            if it:
                pbar.update(it)

            if msg:
                self.send_progress(msg, title=title)

        return (
            preprocess(
                image,
                model=model,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                bbox_padding=bbox_padding,
                on_progress=comfy_progress,
            ),
        )


class SAPIENS_PoseEstimator(SapiensBase):
    """
    Perform pose estimation inference from cropped data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS_MODEL",),
                "image": ("IMAGE",),
                "bboxes": ("SAPIENS_BBOXES",),
                "chunk_size": (
                    "INT",
                    {
                        "default": 48,
                        "max": 96,
                        "min": 1,
                        "tooltip": "Process inputs in chunks of this size",
                    },
                ),
            },
            "optional": {
                "udp_blur": ("INT", {"default": 11, "max": 255, "min": 5}),
                "use_torch_decode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "the non torch decode is much slower but also much more precise",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("SAPIENS_KEYPOINTS", "SAPIENS_BBOXES")
    RETURN_NAMES = ("keypoints", "bboxes")

    OUTPUT_TOOLTIPS = ("Pose keypoints for each frame", "Pass through of the bboxes")

    FUNCTION = "estimate_pose"
    CATEGORY = "SapiensPose"

    @timed_comfy_node("Pose estimation")
    def estimate_pose(
        self,
        *,
        model,
        image: torch.Tensor,
        bboxes,
        chunk_size=16,
        unique_id=None,
        udp_blur=11,
        use_torch_decode=False,
    ):
        if udp_blur % 2 == 0:
            raise ValueError("udp_blur must be an odd number")

        from .support.pipeline import estimate_pose

        self.unique_id = unique_id

        pbar = comfy.utils.ProgressBar(len(bboxes))

        def comfy_progress(it=None, msg=None, title=None):
            if it:
                pbar.update(it)

            if msg:
                self.send_progress(msg, title=title)

        result = estimate_pose(
            model,
            image,
            bboxes,
            chunk_size=chunk_size,
            on_progress=comfy_progress,
            comfy_progress=True,
            udp_blur=udp_blur,
            use_torch=use_torch_decode,
        )

        return (result, bboxes)


class SAPIENS_DrawBones(SapiensBase):
    """ComfyUI Node to draw keypoints and skeletons on an image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "keypoints": ("SAPIENS_KEYPOINTS",),
            },
            "optional": {
                "bboxes": ("SAPIENS_BBOXES",),
                "draw_on_image": ("BOOLEAN", {"default": False}),
                "keypoints_radius": ("INT", {"default": 3}),
                "bone_thickness": ("INT", {"default": 3}),
                "style": (
                    (
                        "sapiens",
                        "openpose",
                    ),
                    {"default": "sapiens"},
                ),
                "keypoints_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Do not draw keypoints below this confidence threshold",
                    },
                ),
                "label_size": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Size of drawn labels",
                    },
                ),
                # toggles
                "draw_face": (
                    ("off", "simple", "clean", "full"),
                    {"default": "clean"},
                ),
                "draw_clavicle": ("BOOLEAN", {"default": False}),
                "draw_upper_body": ("BOOLEAN", {"default": True}),
                "draw_lower_body": ("BOOLEAN", {"default": True}),
                "draw_labels": (("off", "name", "id", "link"), {"default": "off"}),
                "draw_keypoints": ("BOOLEAN", {"default": True}),
                "draw_skeleton": ("BOOLEAN", {"default": True}),
                "draw_bbox": ("BOOLEAN", {"default": False}),
                "bitmask": ("GOLIATH_BITMASK",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw"

    CATEGORY = "SapiensPose"

    @timed_comfy_node("Draw pose")
    def draw(
        self,
        *,
        image,
        keypoints: Result,
        bboxes=None,
        draw_on_image=False,
        keypoints_radius=3,
        bone_thickness=3,
        keypoints_threshold=0.3,
        unique_id=None,
        style: Literal["sapiens", "openpose"] = "sapiens",
        draw_face: Literal["off", "simple", "clean", "full"] = "clean",
        draw_clavicle=False,
        draw_upper_body=True,
        draw_lower_body=True,
        draw_labels="off",
        label_size=0.5,
        draw_keypoints=True,
        draw_skeleton=True,
        draw_bbox=False,
        bitmask=None,
    ):
        self.unique_id = unique_id
        self._msg = ""

        if draw_face not in ["off", "simple", "clean", "full"]:
            draw_face = "clean"

        if draw_labels == "off":
            draw_labels = None

        from .support.pipeline import draw

        output = draw(
            keypoints,
            bboxes_per_frame=bboxes,
            image=image,
            draw_on_image=draw_on_image,
            keypoints_radius=keypoints_radius,
            bone_thickness=bone_thickness,
            keypoints_threshold=keypoints_threshold,
            style=style,
            draw_face=draw_face,
            draw_clavicle=draw_clavicle,
            draw_upper_body=draw_upper_body,
            draw_lower_body=draw_lower_body,
            draw_labels=draw_labels,
            label_size=label_size,
            draw_keypoints=draw_keypoints,
            draw_skeleton=draw_skeleton,
            draw_bbox=draw_bbox,
            bitmask=bitmask,
        )

        try:
            from .support.proxy_tensor import LazyProxyTensor
            if isinstance(output, LazyProxyTensor):
                output = output[:]
        except Exception:
            pass

        return (output,)


# endregion


# region helper nodes
class SAPIENS_PreviewCrops(SapiensBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("SAPIENS_BBOXES",),
            },
            "optional": {
                "color": ("COLOR", {"default": "#3549DE"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"

    CATEGORY = "SapiensPose"
    EXPERIMENTAL = True

    def preview(self, image: torch.Tensor, bboxes: list[np.ndarray], color=None):
        import kornia
        from torch.nn.utils.rnn import pad_sequence

        if color:
            if isinstance(color, str):
                color = tuple(int(color[1 + i : i + 2], 16) for i in (0, 2, 4))

        output = kornia.utils.draw_rectangle(
            image.clone().permute(0, 3, 1, 2),
            pad_sequence([torch.from_numpy(b) for b in bboxes], batch_first=True),
            torch.tensor(color or (0, 255, 0)),
            # torch.tensor((0, 255, 0)),
        )

        return (output.permute(0, 2, 3, 1),)


class SAPIENS_SaveKeypoints(SapiensBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "sapiens_keypoints"}),
                "keypoints": ("SAPIENS_KEYPOINTS",),
                "image": (
                    "IMAGE",
                    {"tooltip": "Image used to extract source dimensions"},
                ),
            },
            "optional": {
                "kind": (("msgpack", "json"), {"default": "msgpack"}),
                "directory": (("output", "input"), {"default": "output"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)

    OUTPUT_NODE = True
    FUNCTION = "save"

    CATEGORY = "SapiensPose"
    EXPERIMENTAL = True

    def save(self, name, keypoints: Result, image, kind, directory):
        import folder_paths
        from .support.io import save_animation

        path = Path(name)
        if path.suffix in [".json", ".msgpack"]:
            kind = path.suffix[1:]

        if not path.is_absolute():
            if directory == "output":
                output_dir = folder_paths.get_output_directory()
            else:
                output_dir = folder_paths.get_input_directory()

            full_output_folder, filename, counter, _subfolder, _filename_prefix = (
                folder_paths.get_save_image_path(path.stem, output_dir)
            )
            full_output = Path(full_output_folder)

            filename_with_batch_num = filename.replace("%batch_num%", "308")

            file = f"{filename_with_batch_num}_{counter:04}{path.suffix}"  # .msgpack"

            output_path = full_output / file
        else:
            output_path = path

        save_animation(output_path, keypoints)

        return (output_path.as_posix(),)


class SAPIENS_SelectKeypointsFrames(SapiensBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("SAPIENS_KEYPOINTS",),
                "start": ("INT", {"default": 0, "max": 999999, "min": 0}),
                "end": ("INT", {"default": -1, "max": 999999, "min": -1}),
                "offset": ("INT", {"default": 0, "max": 999999, "min": 0}),
            }
        }

    RETURN_TYPES = ("SAPIENS_KEYPOINTS",)
    RETURN_NAMES = ("keypoints",)

    FUNCTION = "select"

    CATEGORY = "SapiensPose"
    EXPERIMENTAL = True

    def select(self, keypoints: Result, start, end, offset) -> tuple[Result]:
        return (
            add_metadata(
                keypoints.data[start:end:offset], keypoints.metadata["input_dimensions"]
            ),
        )


class SAPIENS_LoadKeypoints(SapiensBase):
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        files = ["-"] + [
            f.stem
            for f in Path(input_dir).iterdir()
            if f.is_file() and f.name.endswith(".msgpack")
        ]
        return {
            "required": {
                "from_input": (files,),
                "path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("SAPIENS_KEYPOINTS",)
    RETURN_NAMES = ("keypoints",)

    FUNCTION = "load"

    CATEGORY = "SapiensPose"
    EXPERIMENTAL = True

    def load(self, path="", from_input="-"):
        import folder_paths

        if not path and from_input == "-":
            raise ValueError("You must provide a path or select a file from inputs")

        if from_input != "-":
            input_dir = folder_paths.get_input_directory()
            actual_path = Path(input_dir) / f"{from_input}.msgpack"
        else:
            actual_path = Path(path)

        if not actual_path.exists():
            raise RuntimeError(
                f"Path could not be found: {path}, resolved to {actual_path.as_posix()} -> {actual_path.resolve().as_posix()}"
            )

        from .support.io import load_animation

        results = load_animation(actual_path)
        log.info(f"Keypoints metadata: {json.dumps(results.metadata)}")
        return (results,)


class SAPIENS_SmoothKeypoints(SapiensBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("SAPIENS_KEYPOINTS",),
                "window_length": (
                    "INT",
                    {
                        "default": 7,
                        "tooltip": "the chunk of frames to consider for smoothin",
                    },
                ),
                "polyorder": (
                    "INT",
                    {
                        "default": 2,
                        "tooltip": "the order of the polynomial, this is the degree of the fitting polynomial, and must be less than window_length, smaller value smooth more.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SAPIENS_KEYPOINTS",)
    RETURN_NAMES = ("smoothed_keypoints",)
    FUNCTION = "filter"

    CATEGORY = "SapiensPose"
    EXPERIMENTAL = True

    def filter(
        self,
        keypoints,
        window_length=7,
        polyorder=2,
    ):
        from .support.pipeline import smooth_keypoints

        return (
            smooth_keypoints(
                keypoints, window_length=window_length, polyorder=polyorder
            ),
        )


class SAPIENS_FilterBones(SapiensBase):
    @classmethod
    def INPUT_TYPES(cls):
        req = {k: ("BOOLEAN", {"default": True}) for k in GOLIATH_KEYPOINTS}
        return {
            "required": req,
        }

    RETURN_TYPES = ("GOLIATH_BITMASK",)
    RETURN_NAMES = ("bitmask",)
    FUNCTION = "filter"

    CATEGORY = "SapiensPose"
    EXPERIMENTAL = True

    def filter(
        self,
        **kwargs,
    ):
        visibility_list = [kwargs[name] for name in GOLIATH_KEYPOINTS]
        bitmask_tensor = torch.tensor(visibility_list, dtype=torch.bool)
        return (bitmask_tensor,)


# endregion

NODE_CLASS_MAPPINGS = {
    "LoadPoseModel": SAPIENS_LoadPoseModel,
    "SapiensPreprocessor": SAPIENS_Preprocessor,
    "DrawBones": SAPIENS_DrawBones,
    "PoseEstimator": SAPIENS_PoseEstimator,
    "SapiensSaveKeypoints": SAPIENS_SaveKeypoints,
    "SapiensLoadKeypoints": SAPIENS_LoadKeypoints,
    "SapiensSmoothKeypoints": SAPIENS_SmoothKeypoints,
    "SapiensSelectKeypointsFrames": SAPIENS_SelectKeypointsFrames,
    # DEBUG
    "FilterBones": SAPIENS_FilterBones,
    "PreviewCrops": SAPIENS_PreviewCrops,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPoseModel": "Load Sapiens Model",
    "DrawBones": "Sapiens Draw Bones",
    "PoseEstimator": "Sapiens Pose Estimator",
    "SapiensPreprocessor": "Sapiens Preprocessor",
    "SapiensSaveKeypoints": "Sapiens Save Keypoints",
    "SapiensLoadKeypoints": "Load Keypoints",
    "SapiensSmoothKeypoints": "Smooth Keypoints",
    "SapiensSelectKeypointsFrames": "Sapiens Select Keypoints (frames)",
    # DEBUG
    "FilterBones": "Sapiens Filter Bones (Debug)",
    "PreviewCrops": "Preview Crops (Debug)",
}
