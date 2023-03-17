import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F

from mmseg.ops import resize
# from ..builder import HEADS
# from .decode_head import BaseDecodeHead
# from .psp_head import PPM
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.psp_head import PPM
from .contrast_decode_head import BaseDecodeHead

from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from addict import Dict

@HEADS.register_module()
class MaskFormerHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(MaskFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        channels = self.in_channels
        indexs = self.in_index
        backbone_feature_shape = dict()
        for index, channel in zip(indexs, channels):
            backbone_feature_shape[f'res{index+2}'] = Dict({'channel': channel, 'stride': 2**(index+2)})
        print('--------------------------backbone_feature_shape:{}'.format(backbone_feature_shape))
        self.pixel_decoder = self.pixel_decoder_init(backbone_feature_shape)
        self.predictor = self.predictor_init()

    def pixel_decoder_init(self, input_shape):
        common_stride = 4
        transformer_dropout = 0.0
        transformer_nheads = 8
        dim_feedforward = 1024  #1024  2048
        transformer_enc_layers = 4  #4  6
        conv_dim = 256
        mask_dim = 256
        transformer_in_features = ["res3", "res4", "res5"]  # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 transformer_dropout,
                                                 transformer_nheads,
                                                 dim_feedforward,
                                                 transformer_enc_layers,
                                                 conv_dim,
                                                 mask_dim,
                                                 transformer_in_features,
                                                 common_stride)
        return pixel_decoder

    def predictor_init(self):
        in_channels = 256
        num_classes = self.num_classes
        hidden_dim = 256
        num_queries = 100
        nheads = 8
        dim_feedforward = 1024  #2048
        dec_layers = 10 - 1   # 9 decoder layers, add one for the loss on learnable query
        pre_norm = False
        mask_dim = 256
        enforce_input_project = False
        mask_classification = True
        predictor = MultiScaleMaskedTransformerDecoder(in_channels,
                                                       num_classes,
                                                       mask_classification,
                                                       hidden_dim,
                                                       num_queries,
                                                       nheads,
                                                       dim_feedforward,
                                                       dec_layers,
                                                       pre_norm,
                                                       mask_dim,
                                                       enforce_input_project)
        return predictor

    def forward(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions

    def forward_test(self, inputs, img_metas, test_cfg):
        outputs = self.forward(inputs)
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(560, 560),  #(inputs.shape[-2], inputs.shape[-1])
            mode="bilinear",
            align_corners=False,
        )
        del outputs

        pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)

        return pred_masks

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg
