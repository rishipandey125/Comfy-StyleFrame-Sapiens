import gc
import logging
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
import torch.nn.functional as F

from support.proxy_tensor import LazyProxyTensor
from support import pipeline
from support.datatypes import KeypointResult
from support.utils import timed

# for torch.load to be able to unpickle the custom KeypointResult type
__all__ = ["KeypointResult"]

app = typer.Typer(
    add_completion=False,
    help="A CLI for Sapiens-Pose estimation based on the provided playground script.",
    pretty_exceptions_show_locals=False,
)
console = Console()


def save_video(tn: torch.Tensor, output_path: Path = Path("./output.mp4"), fps=24):
    console.print(f"Saving output video to [cyan]{output_path}[/cyan]...")

    from tqdm import tqdm

    pixel_multiplier = 1 if tn.dtype == torch.uint8 else 255

    try:
        with iio.imopen(output_path, "w", plugin="pyav") as writer:
            writer.init_video_stream("libx264", fps=float(int(float(format(fps)))))
            writer.container_metadata["comment"] = "Generated in SapiensPose v1.1"

            for i in tqdm(range(tn.shape[0]), desc="Writting frames to video file"):
                frame = (tn[i] * pixel_multiplier).cpu().numpy()
                writer.write_frame(frame)
    except Exception as e:
        console.print(f"[bold red]Error during video saving:[/bold red] {e}")
        console.print("make sure you have `av`<15 installed")
        console.print(
            "there is currently an issue in >=15.0 https://github.com/imageio/imageio/issues/1139"
        )
        raise typer.Exit(1) from e


# --- Load Video Frames ---
def load_video(
    path,
    *,
    limit: None | int = None,
    offset: None | int = None,
    scale_factor: tuple[float, float] | None = None,
):
    with timed("Loading video"):
        metadata = iio.immeta(path, plugin="pyav")

        fps = metadata["fps"] if metadata["fps"] is not None else 24
        frames_iterator = iio.imiter(path, plugin="pyav")
        frames_list = [frame for frame in frames_iterator]
        frame_count = len(frames_list)

        if offset:
            offset = min(offset, frame_count)
        else:
            offset = 0

        # more then 3 frames here

        end = frame_count
        if limit:
            # warn if less than limit left
            end = min(offset + limit, frame_count)

        frames_tensor = torch.stack(
            [torch.from_numpy(frame) for frame in frames_list[offset:end]]
        )
        if scale_factor:
            frames_tensor = F.interpolate(
                frames_tensor.permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="bilinear",
            ).permute(0, 2, 3, 1)

        # (B, H, W, C) float32 in [0, 1]

        frames_tensor = LazyProxyTensor(frames_tensor)  # .float() / 255.0
        gc.collect()
        console.print(
            f"-> Loaded {frames_tensor.shape[0]} frames at {fps} FPS. ({frames_tensor.shape[2]}x{frames_tensor.shape[1]})"
        )
        return (frames_tensor, fps)


@app.command()
def run(
    # --- Input/Output ---
    input_video: Path = typer.Argument(
        ..., help="Path to the input video file.", exists=True, dir_okay=False
    ),
    output_path: Path = typer.Option(
        "output.mp4",
        "--output",
        "-o",
        help="Path to save the output video or keypoints file.",
    ),
    # --- Workflow Control ---
    load_keypoints: Optional[Path] = typer.Option(
        None,
        help="Load pre-computed keypoints from a .pt file instead of running detection.",
        exists=True,
    ),
    save_keypoints: Optional[Path] = typer.Option(
        None, help="Save the computed keypoints to a .pt file."
    ),
    # --- Model Loading ---
    model_dir: Path = typer.Option(
        Path.home() / ".cache" / "sapiens-pose-cli",
        help="Directory to store/cache downloaded models.",
    ),
    auto_download: bool = typer.Option(
        True,
        "--auto-download/--no-auto-download",
        help="Automatically download models if not found.",
    ),
    device: str = typer.Option(
        "cuda", help="Device to run models on ('cuda' or 'cpu')."
    ),
    # TODO: re-enable Yolo
    bbox_backend: str = typer.Option(
        "RtmDet", help="Bbox detection backend ('RtmDet' or 'Yolov8')."
    ),
    no_compile: bool = typer.Option(
        False, "--no-compile", help="Disable torch.compile for the model."
    ),
    use_torchscript: bool = typer.Option(
        False, "--use-torchscript", help="Use the TorchScript version of the model."
    ),
    debug_detector: bool = typer.Option(
        False,
        "--debug-detector",
        help="Detect bbox and save a video with the detections drawn, returns early in the pipeline",
    ),
    # --- Pre-processing ---
    conf_threshold: float = typer.Option(
        0.5, help="Confidence threshold for bounding box detection."
    ),
    nms_threshold: float = typer.Option(
        0.5, help="Non-Maximum Suppression threshold for bounding box detection."
    ),
    # --- Pose Estimation ---
    chunk_size: int = typer.Option(
        16, help="Batch size for pose estimation processing."
    ),
    # NOTE: broken for now
    # return_heatmaps: bool = typer.Option(
    #     False,
    #     "--return-heatmaps",
    #     help="Output heatmaps instead of keypoints. Drawing will be skipped.",
    # ),
    # --- Drawing ---
    draw_on_black: bool = typer.Option(
        False, help="Draw skeleton on a black background instead of the original video."
    ),
    style: str = typer.Option(
        "sapiens", help="Skeleton drawing style ('sapiens' or 'openpose')."
    ),
    kpt_radius: int = typer.Option(3, help="Radius of the drawn keypoints."),
    bone_thickness: int = typer.Option(
        3, help="Thickness of the drawn skeleton bones."
    ),
    kpt_threshold: float = typer.Option(
        0.3, help="Confidence threshold for drawing keypoints."
    ),
    # --- Drawing Parts ---
    draw_face: bool = typer.Option(True, help="Toggle drawing face keypoints."),
    draw_clavicle: bool = typer.Option(
        False, help="Toggle drawing clavicle keypoints."
    ),
    draw_upper_body: bool = typer.Option(
        True, help="Toggle drawing upper body keypoints."
    ),
    draw_lower_body: bool = typer.Option(
        True, help="Toggle drawing lower body keypoints."
    ),
    draw_labels: bool = typer.Option(
        False, help="Toggle drawing label of keypoints (for debug)."
    ),
    # --- Misc ---
    frame_limit: Optional[int] = typer.Option(
        None,
        "--frame-limit",
        help="Limit processing to the first N frames of the video.",
    ),
    frame_offset: Optional[int] = typer.Option(
        None,
        "--frame-offset",
        help="Skip the first N frames of the video.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
):
    """
    Process a video to perform pose estimation and generate a skeleton overlay.
    """
    logging.basicConfig(
        level="DEBUG" if verbose else "INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # logging.getLogger("support").propagate = False

    console.print(f"Loading video frames from [cyan]{input_video}[/cyan]...")
    try:
        frames_tensor, fps = load_video(
            input_video, limit=frame_limit, offset=frame_offset
        )

    except Exception as e:
        console.print(f"[bold red]Error loading video:[/bold red] {e}")
        raise typer.Exit(1)

    # # temp
    # save_output(frames_tensor)
    # return
    # # temp fin

    keypoints = None

    if load_keypoints:
        console.print(
            f"Loading pre-computed keypoints from [cyan]{load_keypoints}[/cyan]..."
        )
        # weights_only must be False to unpickle the custom KeypointResult class
        keypoints = torch.load(load_keypoints, weights_only=False)
    else:
        # --- Main Pipeline ---
        console.print("[bold green]Starting Pose Estimation Pipeline...[/bold green]")

        model_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Using model directory: [cyan]{model_dir}[/cyan]")

        # 1. Load Model
        model = pipeline.load_model(
            compile=not no_compile,
            use_torchscript=use_torchscript,
            device=device,
            bbox_backend=bbox_backend,
            auto_download=auto_download,
            comfy_progress=False,
            base_path=model_dir,
        )

        # 2. Preprocess
        bboxes_per_frame = pipeline.preprocess(
            frames_tensor,
            model=model,
            confidence_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )
        # if debug_detector:
        #     debug_frames = draw_bbox(frames_tensor, bboxes_per_frame)
        #     save_video(debug_frames, output_path, fps)
        #     console.print(
        #         f"[bold green]Debug Detector complete. Output saved to {output_path}[/bold green]"
        #     )
        #     return

        # 3. Estimate Pose
        pose_data = pipeline.estimate_pose(
            model,
            frames_tensor,
            bboxes_per_frame,
            chunk_size=chunk_size,
            # return_heatmaps=return_heatmaps,
            verbose=verbose,
        )
        if pose_data:
            keypoints, img_bbox_map = pose_data

        else:
            console.print("No pose data found in frames")
            raise typer.Exit(1)

        # if return_heatmaps:
        #     console.print(f"Saving heatmaps to [cyan]{output_path}[/cyan]...")
        #     torch.save(keypoints, output_path)
        #     console.print("[bold green]Done.[/bold green]")
        #     raise typer.Exit()

        if save_keypoints:
            console.print(
                f"Saving computed keypoints to [cyan]{save_keypoints}[/cyan]..."
            )
            torch.save(keypoints, save_keypoints)

    # --- Drawing ---
    if not keypoints:
        console.print(
            "[bold red]No keypoints found or loaded. Cannot proceed to drawing.[/bold red]"
        )
        raise typer.Exit(1)

    console.print("Drawing skeletons on frames...")

    video_height, video_width = frames_tensor.shape[1], frames_tensor.shape[2]

    output_tensor = pipeline.draw(
        (keypoints, img_bbox_map),
        image=frames_tensor,
        draw_on_image=not draw_on_black,
        output_size=(video_height, video_width),
        keypoints_radius=kpt_radius,
        bone_thickness=bone_thickness,
        keypoints_threshold=kpt_threshold,
        style=style,
        verbose=verbose,
        draw_face=draw_face,
        draw_clavicle=draw_clavicle,
        draw_upper_body=draw_upper_body,
        draw_lower_body=draw_lower_body,
        draw_labels=draw_labels,
    )

    # --- Saving Output ---
    save_video(output_tensor, output_path, fps)
    console.print(
        f"[bold green]Processing complete. Output saved to {output_path}[/bold green]"
    )


if __name__ == "__main__":
    app()
