# Copyright (c) OpenMMLab. All rights reserved.
from .mapillary import MapillaryDataset  # noqa: F401,F403
from .pipelines import *  # noqa: F401,F403
from .custom_base import CustomBaseDataset
from .rsipac import RSIPACDataset
__all__ = ['MapillaryDataset', 'CustomBaseDataset', 'RSIPACDataset']   #, 'CustomBaseDataset', 'RSIPACDataset'
