import time
import logging
from ComfyUI_SapiensPose import (
    SAPIENS_LoadPoseModel,
    SAPIENS_PoseSampler,
    SAPIENS_DrawPose,
)
from comfy_mtb.log import log

import proxy


root = proxy.setup()

load_model_node = SAPIENS_LoadPoseModel()
sampler_node = SAPIENS_PoseSampler()
draw_node = SAPIENS_DrawPose()

# method = getattr(load_model_node, load_model_node.FUNCTION)
# print(method)

start_total = time.perf_counter()


(model,) = load_model_node.load_model()

demo = root / "woman-running-sample_image_00001.pt"

img = torch.load(demo.as_posix(), map_location="cuda")

start = time.perf_counter()
log.info("Starting kpt sampling")
(res, keypoints, bboxes) = sampler_node.process(
    model=model, image=img, bbox_padding=0, batch_size=32, unique_id=None
)
log.info(f"Sampling finished in {time.perf_counter() - start:.4f} seconds")

start = time.perf_counter()

log.info("Starting drawing")
(out,) = draw_node.draw(res, keypoints, bboxes)
logging.info(f"Drawing finished in {time.perf_counter() - start:.4f} seconds")

log.info(f"Total time: {time.perf_counter() - start_total:.4f} seconds")
# (model,) = sapiens_nodes.SAPIENS_LoadPoseModel()
# print(dir(sapiens_nodes))

log.info("saving video...")
write_video("output.mp4", out)

log.info("video saved!")
