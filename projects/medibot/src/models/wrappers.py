from typing import Literal, NamedTuple
import warnings
import timm
from torch import nn
import torch


class ResnetWrapper(nn.Module):
    """Wraps resnet to return last layer feature map as ``patch tokens`` and avgpool as ``class token``"""

    def __init__(self, model: timm.models.ResNet):
        super().__init__()
        self.model = model

    def forward(self, x, mask=None, return_all_tokens=False):
        feature_map = self.model.forward_features(x)
        b, c, h, w = feature_map.shape
        avgpool = self.model.global_pool(feature_map)

        if not return_all_tokens: 
            return avgpool
            
        feature_map_as_tokens = feature_map.reshape(b, c, h * w).permute(0, 2, 1)
        b, c = avgpool.shape
        avgpool = avgpool[:, None, :]

        return torch.cat([avgpool, feature_map_as_tokens], dim=1)

    def get_class_token(self, x) -> torch.Tensor:
        return self(x)

    def get_feature_map(self, x) -> torch.Tensor:
        patch_tokens = self(x, return_all_tokens=True)[:, 1:, :]
        return self._tokens_to_feature_map(patch_tokens)
    

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(
        self,
        backbone,
        head=None,
    ):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone

        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx:end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx:end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_


def forward_multi_crop(model, x, mask=None, **kwargs):

    if not isinstance(x, list):
        x = [x]
        mask = [mask] if mask is not None else None

    idx_crops = torch.cumsum(
        torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1],
        0,
    )

    start_idx = 0
    for end_idx in idx_crops:
        inp_x = torch.cat(x[start_idx:end_idx])

        if mask is not None:
            inp_m = torch.cat(mask[start_idx:end_idx])
            kwargs.update(dict(mask=inp_m))
        
        _out = model(inp_x, **kwargs)
        if start_idx == 0:
            outputs = _out
        else: 
            outputs = torch.cat((outputs, _out))
        start_idx = end_idx

    return outputs


class ViTOutput(NamedTuple):
    cls_token: torch.Tensor
    extra_tokens: torch.Tensor | None
    patch_tokens: torch.Tensor | None


class ViTWrapper(nn.Module): 
    def __init__(self, model: nn.Module): 
        super().__init__()
        self.model = model

    def forward(self, x, mask=None): 
        #outputs = self.model(x, mask=mask, return_all_tokens=True)
        if isinstance(x, list): 
            n_images = len(x)
        else: 
            n_images = 1

        outputs = forward_multi_crop(self.model, x, mask=mask, return_all_tokens=True)

        n_cls_tokens = self.model.n_cls_tokens
        cls_token = outputs[:, 0, :]
        extra_tokens = outputs[:, 1:n_cls_tokens, :]
        patch_tokens = outputs[:, n_cls_tokens:, :]
    
        return ViTOutput(cls_token, extra_tokens, patch_tokens)