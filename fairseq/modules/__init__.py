# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""
from .efficient_pa_attention_qk import EfficientPAQKMultiheadAttention
from .efficient_pa_attention_conv import EfficientPAConvMultiheadAttention
from .efficient_pa_attention import UniversalEfficientPaAttention
from .efficient_pa_attention_efficient_FFN import UniversalEfficientPaAttentionFFN
from .cross_efficient_pa_attention_efficient_FFN import MultiheadAttentionFFN
from .cross_efficient_pa_attention_efficient_FFN_conv import MultiheadAttentionFFNConv
from .universal_dynamic_graphstructure_generation import UniversalLightWeightDynamicGraphStructureGeneration,UniversalRelativeDynamicGraphStructureGeneration,UniversalDynamicGraphStructureGeneration,UniversalStandardDynamicGraphStructureGeneration
from .lp_multihead_attention import LPMultiheadAttention

from .efficient_pa_attention_qk_wo_PGFFNs import EfficientPAQKMultiheadAttentionWOPGFFNs
from .linear import Linear
from .adaptive_input import AdaptiveInput
from .adaptive_softmax import AdaptiveSoftmax
from .AFFN_BLOCK import AFFNBlock
from .Relative_AFFN_BLOCK import RelativeAFFNBlock
from .beamable_mm import BeamableMM
from .character_token_embedder import CharacterTokenEmbedder
from .conv_tbc import ConvTBC
from .cross_entropy import cross_entropy
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from .dynamic_convolution import DynamicConv, DynamicConv1dTBC
from .refiner_multihead_attention import RefinerrMultiheadAttention 
from .dynamic_crf_layer import DynamicCRF
from .fairseq_dropout import FairseqDropout
from .fp32_group_norm import Fp32GroupNorm
from .gelu import gelu, gelu_accurate
from .grad_multiply import GradMultiply
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .kmeans_vector_quantizer import KmeansVectorQuantizer
from .layer_drop import LayerDropModuleList
from .layer_norm import Fp32LayerNorm, LayerNorm
from .learned_positional_embedding import LearnedPositionalEmbedding
from .lightweight_convolution import LightweightConv, LightweightConv1dTBC
from .linearized_convolution import LinearizedConvolution
from .multihead_attention import MultiheadAttention, MultiheadAttention820
from .relative_multihead_attention import RelativeMultiheadAttention
from .positional_embedding import PositionalEmbedding
from .same_pad import SamePad
from .scalar_bias import ScalarBias
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .transformer_sentence_encoder import TransformerSentenceEncoder
from .transpose_last import TransposeLast
from .unfold import unfold1d
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer, TransformerMultiBranchEncoderLayer, TransformerReAttentionEncoderLayer, TransformerEncoderLayer_layer_level_clean
from .vggblock import VGGBlock
from .transformer_layer import NormformerDecoderLayer, NormformerEncoderLayer, TransformerUpSampleEncoderLayer, TransformerEncoderLayer_layer_level, MultiScaleOddEvenTransformerEncoderLayer
from .transformer_layer import TransformerBranchEncoderLayer, TransformerMBEncoderLayer, PATransformerEncoderLayerV2, PATransformerDecoderLayerV2
from .refiner_attention import RefinerAttention
from .graphstruture_learning import DynamicGraphStructureMultiheadAttention,DynamicGraphStructureGeneration, DynamicGraphStructureMultiheadAttentionAttnscale, DynamicGraphStructureEnhancedMultiheadAttention,DynamicGraphStructureMultiheadAttentionLoopEnhancedAfter,DynamicGraphStructureMultiheadAttentionLoopEnhancedBefore,DynamicGraphStructureMultiheadAttentionPure,DynamicGraphStructureMultiheadAttentionRefiner
from .relative_multihead_reattention import RelativeMultiheadReAttention
from .scale_multihead_attention import ScaleMultiheadAttention
from .pa_transformer_layer import PADecoderSANGlobalAGenLayer, PATransformerDecoderLayer,PATransformerEncoderLayer
from .scale_relative_multihead_attention import ScaleRelativeMultiheadAttention
from .light_transformer_layer import LightTransformerEncoderLayer, LightTransformerDecoderLayer


__all__ = [
    "LPMultiheadAttention",
    "AdaptiveInput",
    "Linear",
    "MultiheadAttention820",
    "EfficientPAQKMultiheadAttention",
    "EfficientPAConvMultiheadAttention",
    "UniversalEfficientPaAttention",
    "UniversalEfficientPaAttentionFFN"
    "MultiheadAttentionFFN",
    "MultiheadAttentionFFNConv",
    "RelativeMultiheadAttentionFFN",
    "UniversalLightWeightDynamicGraphStructureGeneration",
    "UniversalRelativeDynamicGraphStructureGeneration",
    "UniversalDynamicGraphStructureGeneration",
    "UniversalStandardDynamicGraphStructureGeneration"
    "EfficientPAQKMultiheadAttentionWOPGFFNs",
    "PADecoderSANGlobalAGenLayer",
    "TransformerBranchEncoderLayer",
    "TransformerMBEncoderLayer",
    "PATransformerDecoderLayer",
    "PATransformerEncoderLayer",
    "PATransformerEncoderLayerV2",
    "PATransformerDecoderLayerV2",
    "PartialMacronFormerBlcok",
    "TalkingheadAttention",
    "RelativeAFFNBlock",
    "AFFNBlock",
    "LightTransformerEncoderLayer",
    "LightTransformerDecoderLayer",
    "ELSMultiheadAttention",
    "AdaptiveSoftmax",
    "RefinerAttention",
    "BeamableMM",
    "TransformerEncoderLayer_layer_level",
    "RefinerrMultiheadAttention",
    "TransformerUpSampleEncoderLayer",
    "ScaleMultiheadAttention",
    "CharacterTokenEmbedder",
    "ConvTBC",
    "cross_entropy",
    "DownsampledMultiHeadAttention",
    "DynamicConv1dTBC",
    "DynamicConv",
    "DynamicCRF",
    "FairseqDropout",
    "Fp32GroupNorm",
    "Fp32LayerNorm",
    "NormformerEncoderLayer",
    "NormformerDecoderLayer",
    "gelu",
    "gelu_accurate",
    "GradMultiply",
    "GumbelVectorQuantizer",
    "KmeansVectorQuantizer",
    "LayerDropModuleList",
    "LayerNorm",
    "TransformerEncoderLayer_layer_level_clean",
    "LearnedPositionalEmbedding",
    "LightweightConv1dTBC",
    "LightweightConv",
    "LinearizedConvolution",
    "MultiheadAttention",
    "PositionalEmbedding",
    "RelativeMultiheadReAttention",
    "SamePad",
    "ScalarBias",
    "SinusoidalPositionalEmbedding",
    "TransformerSentenceEncoderLayer",
    "TransformerSentenceEncoder",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "TransformerMultiBranchEncoderLayer",
    "TransformerReAttentionEncoderLayer",
    "DynamicGraphStructureMultiheadAttention",
    "DynamicGraphStructureMultiheadAttentionAttnscale",
    "DynamicGraphStructureEnhancedMultiheadAttention",
    "DynamicGraphStructureMultiheadAttentionLoopEnhancedAfter",
    "DynamicGraphStructureMultiheadAttentionLoopEnhancedBefore",
    "DynamicGraphStructureMultiheadAttentionPure",
    "DynamicGraphStructureMultiheadAttentionRefiner",
    "DynamicGraphStructureGeneration",
    "TransposeLast",
    "VGGBlock",
    "unfold1d",
]
