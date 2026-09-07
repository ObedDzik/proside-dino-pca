from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
import json
import os
from typing import Literal

import numpy as np
import sklearn
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from tqdm import tqdm
import copy
from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
from medAI.transforms.crop_to_mask import CropToMask
from medAI.transforms.pixel_augmentations import RandomContrast, RandomGamma
from medAI.datasets.nct2013.cohort_selection import (
    get_parser as get_cohort_selection_parser,
    select_cohort_from_args,
)
from medAI.datasets.nct2013.data_access import data_accessor
from medAI.transforms.prostnfound_transform import ProstNFoundTransform
from typing import List, Optional
import random


def get_dataloaders_from_args(args, mode: Literal["train", "test", "heatmap"] = "train"):
    choose_params = args.get('choose_params', False)


    if args.flip_ud:
        transform_flip_ud = True
    else: 
        transform_flip_ud = False
    case_ids = None

    #TODO: The prostate mask is turned on and off in the prostnfoundtransform class
    train_transform = ProstNFoundTransform(
        augment=args.augmentations,
        image_size=args.image_size,
        mask_size=args.mask_size,
        mean=args.mean,
        std=args.std,
        crop_to_prostate=args.crop_to_prostate,
        first_downsample_size=args.first_downsample_size,
        return_raw_images=mode != "train",
        grade_group_for_positive_label=vars(args).get(
            "grade_group_for_positive_label", 1
        ),
        flip_ud=transform_flip_ud,
    )
    val_transform = ProstNFoundTransform(
        augment="none",
        image_size=args.image_size,
        mask_size=args.mask_size,
        mean=args.mean,
        std=args.std,
        crop_to_prostate=args.crop_to_prostate,
        first_downsample_size=args.first_downsample_size,
        return_raw_images=mode != "train",
        grade_group_for_positive_label=vars(args).get(
            "grade_group_for_positive_label", 1
        ),
        flip_ud=transform_flip_ud,
    )
        
    train_cores, val_cores, test_cores = select_cohort_from_args(args) #train only, the rest are empty!
    train_dataset = BModeDatasetV1(
        train_cores,
        train_transform,
        rf_as_bmode=args.rf_as_bmode,
        include_rf=args.include_rf,
        flip_ud=args.flip_ud,
        frames=args.frames,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    nct_val_dataset = BModeDatasetV1(
        val_cores,
        val_transform,
        rf_as_bmode=args.rf_as_bmode,
        include_rf=args.include_rf,
        flip_ud=args.flip_ud,
        frames=args.frames,
    )

    nct_val_loader = DataLoader(
        nct_val_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return(dict(train=train_loader, val=nct_val_loader))
