from argparse import Namespace
from logging import warn

from medAI.modeling.vision_transformer import VisionTransformer
from .patch_embed import PatchEmbed, FFTPatchEmbed, FFTPatchEmbedV2
from medAI.modeling.swin_transformer import swin_tiny, swin_small, swin_base, swin_large
import torch 
from torch import nn
#from . import vision_transformer as vits
#from torchvision import models as torchvision_models


MODEL_REGISTRY = {}


def register_model(func): 
    MODEL_REGISTRY[func.__name__] = func
    return func


def get_model(name, **kwargs):
    return MODEL_REGISTRY[name](**kwargs)


@register_model
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


@register_model
def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


@register_model
def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


@register_model
def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


@register_model
def vit_small_fft_v2(patch_size=16, **kwargs):
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbedV2, **kwargs)


@register_model
def vit_small_fft_v1(patch_size=16, **kwargs): 
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbed, **kwargs)  


@register_model
def dinov2_vitb14(**kwargs):
    for k in kwargs.keys(): 
        warn(f"Unused argument {k}")

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    return model


def _medsam(**kwargs): 
    from .medsam import MedSAMIBot
    model = MedSAMIBot(**kwargs)
    model.embed_dim = 768
    return model


@register_model
def medsam_ibot(**kwargs): 
    from .medsam import MedSAMIBot
    kwargs["version"] = "2"
    model = MedSAMIBot(**kwargs)
    model.embed_dim = 768
    return model


def _resnet(name, **kwargs):
    from timm.models import resnet
    model = resnet.__dict__[name](**kwargs)
    embed_dim = model.fc.weight.shape[1]
    model.fc = nn.Identity()
    from .wrappers import ResnetWrapper
    model = ResnetWrapper(model)
    model.embed_dim = embed_dim
    return model


@register_model
def resnet18(**kwargs):
    return _resnet('resnet18', **kwargs)


@register_model
def resnet34(**kwargs):
    return _resnet('resnet34', **kwargs)


@register_model
def resnet50(**kwargs):
    return _resnet('resnet50', **kwargs)


@register_model
def ibot_teacher(**kwargs):
    conf = Namespace(**kwargs)

    student: nn.Module
    teacher: nn.Module

    # ============ building student and teacher networks ... ============
    kw = {}
    kw.update(conf.get("backbone_kw", {}))
    logging.info(f"Model kwargs: {kw}")

    # we changed the name DeiT-S for ViT-S to avoid confusions
    conf.arch = conf.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if conf.arch == "medsam":
        student = MedSAMIBot(**kw)
        teacher = MedSAMIBot(**kw)
        embed_dim = 768

    elif conf.arch in models.__dict__.keys() and "swin" in conf.arch:
        student = models.__dict__[conf.arch](
            window_size=conf.window_size,
            masked_im_modeling=conf.use_masked_im_modeling,
            **kw,
        )
        teacher = models.__dict__[conf.arch](
            window_size=conf.window_size,
            drop_path_rate=0.0,
            **kw,
        )
        embed_dim = student.num_features

    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif conf.arch in MODEL_REGISTRY.keys():
        student = MODEL_REGISTRY[conf.arch](
            patch_size=conf.patch_size,
            drop_path_rate=conf.drop_path,
            masked_im_modeling=conf.use_masked_im_modeling,
            **kw,
        )
        teacher = MODEL_REGISTRY[conf.arch](patch_size=conf.patch_size, **kw)
        embed_dim = student.embed_dim
    elif "resnet" in conf.arch:
        student = resnet.__dict__[conf.arch]()
        teacher = resnet.__dict__[conf.arch]()
        embed_dim = student.fc.weight.shape[1]
        student.fc = nn.Identity()
        teacher.fc = nn.Identity()
        student = src.models.wrappers.ResnetWrapper(student)
        teacher = src.models.wrappers.ResnetWrapper(teacher)
    else:
        logging.info(f"Unknow architecture: {conf.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = src.models.wrappers.MultiCropWrapper(
        student,
        iBOTHead(
            embed_dim,
            conf.out_dim,
            patch_out_dim=conf.patch_out_dim,
            norm=conf.norm_in_head,
            act=conf.act_in_head,
            norm_last_layer=conf.norm_last_layer,
            shared_head=conf.shared_head,
        ),
    )
    teacher = src.models.wrappers.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim,
            conf.out_dim,
            patch_out_dim=conf.patch_out_dim,
            norm=conf.norm_in_head,
            act=conf.act_in_head,
            shared_head=conf.shared_head_teacher,
        ),
    )
    return student, teacher


@register_model 
def ibot_student(): 
    ...


