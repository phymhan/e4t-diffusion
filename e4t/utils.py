import os
import json
from requests.exceptions import HTTPError
import huggingface_hub
from huggingface_hub.utils._errors import EntryNotFoundError

import numpy as np
import albumentations
from PIL import Image
import torch
from diffusers.utils import load_image as load_image_diffusers
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from e4t.models.unet_2d_condition import UNet2DConditionModel
from e4t.encoder import E4TEncoder
from einops import rearrange, repeat
import torch.nn as nn


class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()


def download_from_huggingface(repo, filename, **kwargs):
    while True:
        try:
            return huggingface_hub.hf_hub_download(
                repo,
                filename=filename,
                **kwargs
            )
        except HTTPError as e:
            if e.response.status_code == 401:
                # Need to log into huggingface api
                huggingface_hub.interpreter_login()
                continue
            elif e.response.status_code == 403:
                # Need to do the click through license thing
                print(
                    f"Go here and agree to the click through license on your account: https://huggingface.co/{repo}"
                )
                input("Hit enter when ready:")
                continue
            else:
                raise e


MODELS = {
    "e4t-diffusion-ffhq-celebahq-v1": {
        "repo": "mshing/e4t-diffusion-ffhq-celebahq-v1",
        "subfolder": None,
    }
}
FILES = ["weight_offsets.pt", "encoder.pt", "config.json"]


def load_config_from_pretrained(pretrained_model_name_or_path):
    if os.path.exists(pretrained_model_name_or_path):
        if "config.json" not in pretrained_model_name_or_path:
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, "config.json")
    else:
        assert pretrained_model_name_or_path in MODELS, f"Choose from {list(MODELS.keys())}"
        pretrained_model_name_or_path = download_from_huggingface(
            repo=MODELS[pretrained_model_name_or_path]["repo"],
            filename="config.json",
            subfolder=MODELS[pretrained_model_name_or_path]["subfolder"]
        )
    with open(pretrained_model_name_or_path, "r", encoding="utf-8") as f:
        pretrained_args = AttributeDict(json.load(f))
    return pretrained_args


def load_e4t_unet(pretrained_model_name_or_path=None, ckpt_path=None, **kwargs):
    assert pretrained_model_name_or_path is not None or ckpt_path is not None
    if pretrained_model_name_or_path is None or (ckpt_path is not None and not os.path.exists(ckpt_path)):
        if os.path.exists(ckpt_path):
            assert os.path.basename(ckpt_path) == "unet.pt" or os.path.basename(ckpt_path) == "weight_offsets.pt", "You must specify the filename! (`unet.pt` or `weight_offsets.pt`)"
            config = load_config_from_pretrained(os.path.dirname(ckpt_path))
        else:
            assert ckpt_path in MODELS, f"Choose from {list(MODELS.keys())}"
            config = load_config_from_pretrained(ckpt_path)
            try:
                ckpt_path = download_from_huggingface(
                    repo=MODELS[ckpt_path]["repo"],
                    filename="weight_offsets.pt",
                    subfolder=MODELS[ckpt_path]["subfolder"]
                )
            except EntryNotFoundError:
                ckpt_path = download_from_huggingface(
                    repo=MODELS[ckpt_path]["repo"],
                    filename="unet.pt",
                    subfolder=MODELS[ckpt_path]["subfolder"]
                )
        pretrained_model_name_or_path = config.pretrained_model_name_or_path if config.pretrained_args is None else config.pretrained_args["pretrained_model_name_or_path"]
    unet = OriginalUNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", **kwargs)
    state_dict = dict(unet.state_dict())
    if ckpt_path:
        ckpt_sd = torch.load(ckpt_path, map_location="cpu")
        state_dict.update(ckpt_sd)
        print(f"Resuming from {ckpt_path}")
    unet = UNet2DConditionModel(**unet.config)
    m, u = unet.load_state_dict(state_dict, strict=False)
    if len(m) > 0 and ckpt_path:
        raise RuntimeError(f"missing keys:\n{m}")
    if len(u) > 0:
        raise RuntimeError(f"unexpected keys:\n{u}")
    return unet


def save_e4t_unet(model, save_dir):
    weight_offsets_sd = {k: v for k, v in model.state_dict().items() if "wo" in k}
    torch.save(weight_offsets_sd, os.path.join(save_dir, "weight_offsets.pt"))


def load_e4t_encoder(ckpt_path=None, **kwargs):
    encoder = E4TEncoder(**kwargs)
    if ckpt_path:
        if os.path.exists(ckpt_path):
            if "encoder.pt" not in ckpt_path:
                ckpt_path = os.path.join(ckpt_path, "encoder.pt")
        else:
            assert ckpt_path in MODELS, f"Choose from {list(MODELS.keys())}"
            ckpt_path = download_from_huggingface(
                repo=MODELS[ckpt_path]["repo"],
                filename="encoder.pt",
                subfolder=MODELS[ckpt_path]["subfolder"]
            )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        print(f"Resuming from {ckpt_path}")
        m, u = encoder.load_state_dict(state_dict, strict=False)
        if len(m) > 0:
            raise RuntimeError(f"missing keys:\n{m}")
        if len(u) > 0:
            raise RuntimeError(f"unexpected keys:\n{u}")

    return encoder


def save_e4t_encoder(model, save_dir):
    torch.save(model.state_dict(), os.path.join(save_dir, "encoder.pt"))


def make_transforms(size, random_crop=False):
    rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=3)
    if not random_crop:
        cropper = albumentations.CenterCrop(height=size, width=size)
    else:
        cropper = albumentations.RandomCrop(height=size, width=size)
    return albumentations.Compose([rescaler, cropper])


def load_image(image_path, resolution=None):
    pil_image = load_image_diffusers(image_path)
    if resolution:
        processor = make_transforms(resolution)
        image = np.array(pil_image)
        image = processor(image=image)["image"]
        pil_image = Image.fromarray(image)
    return pil_image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class EmbedPointer:
    def __init__(self, embed_pointer=None):
        self.embed_pointer = embed_pointer
        self.embed = None
        self.cross_attn_count = 0
        self.global_step = -1

    def __call__(self):
        return self.embed_pointer


def regiter_attention_editor_diffusers(model, editor: EmbedPointer):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def wo_forward(self, place_in_unet):
        def forward(input: torch.Tensor, **kwargs):
            embed_y = editor.embed_pointer[0][1]
            if embed_y is None:
                return 0
            # embed_y = editor.embed_pointer[0][1]
            input_x = input.mean(dim=1)
            # if editor.global_step != editor.embed_pointer[0][0]:
            #     breakpoint()
            #     editor.global_step = editor.embed_pointer[0][0]
            # input_x = torch.cat([input_x, self.v.unsqueeze(0)], dim=1)
            vx = self.linear1(input_x.detach()) # (row_dim)
            vy = self.linear2(embed_y) # (column_dim)
            # vx = self.linear1(self.v) # (row_dim)
            # vy = self.linear2(self.v) # (column_dim)
            # matrix multiplication -> (row_dim, column_dim)
            # v_matrix = vx.unsqueeze(0).T * vy.unsqueeze(0)
            v_matrix = torch.einsum('b i, b j -> b i j', vx, vy)
            # columnwise
            # v_matrix = self.linear_column(v_matrix.T)
            v_matrix = self.linear_column(rearrange(v_matrix, 'b i j -> b j i'))
            # rowwise
            # v_matrix = self.linear_row(v_matrix.T)
            v_matrix = self.linear_row(rearrange(v_matrix, 'b i j -> b j i'))
            # if self.row_dim != self.column_dim:
            #     breakpoint()
            # return v_matrix.T
            return rearrange(v_matrix, 'b i j -> b j i')

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'WeightOffsets':  # spatial Transformer layer
                net.forward = wo_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count
