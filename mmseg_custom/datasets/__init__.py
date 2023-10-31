# Copyright (c) OpenMMLab. All rights reserved.
from .mapillary import MapillaryDataset  # noqa: F401,F403
from .pipelines import *  # noqa: F401,F403
from .custom_base import CustomBaseDataset
from .rsipac import RSIPACDataset
from .loveda import LOVEDADataset
from .uda_dataset import UDADataset
from .gta import GTADataset
from .sfda_dataset import SFDADataset
from .pseudo_cityscapes import PseudoCityscapesDataset
__all__ = ['MapillaryDataset',
           'CustomBaseDataset',
           'RSIPACDataset',
           'LOVEDADataset',
           'GTADataset',
           'UDADataset',
           'SFDADataset',
           'PseudoCityscapesDataset']   #, 'CustomBaseDataset', 'RSIPACDataset'
