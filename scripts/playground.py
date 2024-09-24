from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules.script_callbacks import on_script_unloaded
from modules.ui_components import InputAccordion
from modules import scripts

from scripts.pg_xyz import xyz_support

from functools import wraps
import gradio as gr
import torch


original_callback = KDiffusionSampler.callback_state


@torch.inference_mode()
@wraps(original_callback)
def pg_callback(self, d):

    params: dict = getattr(self, "playground", {})
    enable: bool = params.get("enable", False)
    chn: int = params.get("chn", 1)
    delta: float = params.get("delta", 0.0)

    if not enable:
        return original_callback(self, d)

    X: torch.Tensor = d["x"].detach().clone()
    batchSize: int = X.size(0)
    Y = torch.ones_like(X)

    for b in range(batchSize):
        d["x"][b][chn] += Y[b][chn] * delta

    return original_callback(self, d)


KDiffusionSampler.callback_state = pg_callback


class Playground(scripts.Script):

    def __init__(self):
        self.xyz: dict = {}
        xyz_support(self.xyz)

    def title(self):
        return "Latent Playground"

    def show(self, is_img2img):
        return None if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):

        with InputAccordion(False, label=self.title()) as enable:
            with gr.Row():
                chn = gr.Number(value=0, label="Channel")
                delta = gr.Number(value=0.0, label="Delta")
            chn.do_not_save_to_config = True
            delta.do_not_save_to_config = True

        return enable, chn, delta

    def process(
        self,
        p,
        enable: bool,
        chn: int,
        delta: float,
    ):

        chn = int(self.xyz.get("chn", chn)) - 1
        delta = float(self.xyz.get("delta", delta))
        delta /= getattr(p, "steps", 20.0)
        KDiffusionSampler.playground = {"enable": enable, "chn": chn, "delta": delta}


def restore_callback():
    KDiffusionSampler.callback_state = original_callback


on_script_unloaded(restore_callback)
