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
from .pseudo_isprs import PseudoISPRSDataset
from .potsdam import PotsdamDataset
from .isprs import IsprsDataset
from .dfc22 import DFCDataset
from .pseudo_loveda import PseudoLoveDataset

__all__ = ['MapillaryDataset',
           'CustomBaseDataset',
           'RSIPACDataset',
           'LOVEDADataset',
           'GTADataset',
           'UDADataset',
           'SFDADataset',
           'PseudoCityscapesDataset',
           'PseudoISPRSDataset',
           'PotsdamDataset',
           'IsprsDataset',
           'DFCDataset',
           'PseudoLoveDataset']   #, 'CustomBaseDataset', 'RSIPACDataset'
