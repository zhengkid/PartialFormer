# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules.scale_multihead_attention import ScaleMultiheadAttention
from fairseq.modules.scale_relative_multihead_attention import ScaleRelativeMultiheadAttention
from fairseq.modules import LayerNorm, MultiheadAttention, RelativeMultiheadAttention,MultiheadAttentionFFNConv, UniversalStandardDynamicGraphStructureGeneration, UniversalDynamicGraphStructureGeneration ,MultiheadAttentionFFN,EfficientPAQKMultiheadAttention,EfficientPAConvMultiheadAttention
from fairseq.modules.relative_multihead_reattention import RelativeMultiheadReAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
import math
from torch import Tensor

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.bath_num = 0
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.args = args
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
        )
        else:
            return RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        self.bath_num += 1
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, atten_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        # if view_atten_mapp:
        #     task = tasks.setup_task(args)
        #     src_dict = getattr(task, "source_dictionary", None)
        #     bsz, src_len = atten_weights.size(0), atten_weights.size(1)
        #     atten_weights = atten_weights.numpy()
        #     for i in range(bsz):
        #         atten_i = atten_weights[i]
        #         draw(atten_i, src_dict, "batch-"+str(self.bath_num)+"sentence-"+str(i), "/mnt/libei/atten-map/batch-"+str(self.bath_num)+"sentence-"+str(i)+".jpg")
            
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class PATransformerDecoderLayerV2(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True

        self.onnx_trace = False

    def build_self_attention(
            self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        if args.max_relative_length == -1:
            if args.macaron:
                return PartialMacronFormerBlcok(
                args,
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                encoder=False,
                )
            else:
                return DecoderChannelBasedPartialFormerBlcok(
                    args,
                    embed_dim,
                    args.decoder_attention_heads,
                    dropout=args.attention_dropout,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    self_attention=not getattr(args, "cross_self_attention", False),
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                )
        else:

            return RelativeMultiheadAttentionFFN(args,
                                                 self.embed_dim, args.decoder_attention_heads,
                                                 args.max_relative_length, dropout=args.attention_dropout,
                                                 k_only=args.k_only,
                                                 )

    def build_encoder_attention(self, embed_dim, args):
        if args.macaron:
            return PartialMacronFormerBlcok(
                args,
                embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                encoder=False,
            )
        else:
            return DecoderChannelBasedPartialFormerBlcok(
                args,
                embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
            self,
            x,
            attn_before: Optional[torch.Tensor] = None,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
            attn_before=attn_before,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def compute_macs_params(self, T=1, S=1):
        macs = 0
        n_params = 0
        macs_attn = 0

        # LayerNorm
        n_params += sum([p.numel() for p in self.self_attn_layer_norm.parameters()])
        # n_params += sum([p.numel() for p in self.final_layer_norm.parameters()])

        # self attention
        self_attn_layer = self.self_attn.compute_macs_params(T=T, S=T)
        macs += self_attn_layer['macs']
        n_params += self_attn_layer['params']
        macs_attn += self_attn_layer['macs_attn']

        # Encoder-decoder attn
        if self.encoder_attn is not None:
            # self attention scaled-dot-product Attn
            enc_attn = self.encoder_attn.compute_macs_params(T=T, S=S)
            macs += enc_attn['macs']
            n_params += enc_attn['params']
            macs_attn += enc_attn['macs_attn']

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': macs_attn
        }

class PATransformerEncoderLayerV2(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.args = args

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            if args.macaron:
                return PartialMacronFormerBlcok(
                    args,
                    embed_dim,
                    args.encoder_attention_heads,
                    dropout=args.attention_dropout,
                    self_attention=True,
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                )
            else:
                return ChannelBasedPartialFormerBlcok(
                args,
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                )
        else:
            return RelativeEfficientPAQKMultiheadAttention(args,
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None, attn_before: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, atten_weights  = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            attn_before = attn_before,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        return x, atten_weights

        # return x, atten_weights, local_attn_maps, features_before

    def compute_macs_params(self, S=1):
        macs = 0
        n_params = 0
        macs_attn = 0

        # Layer Norms
        # MACS are zero for LayerNorm because they can be fused
        n_params += sum([p.numel() for p in self.self_attn_layer_norm.parameters()])
        # n_params += sum([p.numel() for p in self.final_layer_norm.parameters()])

        # Attn
        self_attn_layer = self.self_attn.compute_macs_params(T=S, S=S)
        macs += self_attn_layer['macs']
        n_params += self_attn_layer['params']
        macs_attn += self_attn_layer['macs_attn']

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': macs_attn
        }




class TransformerMBEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = int(args.encoder_embed_dim / args.num_branch)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            int(args.encoder_ffn_embed_dim / args.num_branch),
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            int(args.encoder_ffn_embed_dim / args.num_branch),
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.args = args
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, atten_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, atten_weights


class TransformerBranchEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()

        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.args = args
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x

        if self.normalize_before:
            x_enhance = self.final_layer_norm(x)
        x_enhance = self.activation_fn(self.fc1(x_enhance))
        x_enhance = self.activation_dropout_module(x_enhance)
        x_enhance = self.fc2(x_enhance)
        x_enhance = self.dropout_module(x_enhance)

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, atten_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        x = self.dropout_module(x)
        x = x + x_enhance
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        return x


class PATransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.args = args

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return EfficientPAQKMultiheadAttention(
                args,
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None,
                attn_before: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, atten_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            attn_before=attn_before,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        return x, atten_weights



    def compute_macs_params(self, S=1):
        macs = 0
        n_params = 0
        macs_attn = 0

        # Layer Norms
        # MACS are zero for LayerNorm because they can be fused
        n_params += sum([p.numel() for p in self.self_attn_layer_norm.parameters()])
        # n_params += sum([p.numel() for p in self.final_layer_norm.parameters()])

        # Attn
        self_attn_layer = self.self_attn.compute_macs_params(T=S, S=S)
        macs += self_attn_layer['macs']
        n_params += self_attn_layer['params']
        macs_attn += self_attn_layer['macs_attn']

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': macs_attn
        }


class MultiScaleOddEvenTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
        )
        else:
            return MultiScaleRelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )
    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, src_token=None, layer_num=None, batch_num=None,attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, atten_weight = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        '''
        if layer_num==0:
            task = tasks.setup_task(self.args)
            src_dict = getattr(task, "source_dictionary", None)
            bsz, src_len = atten_weight.size(0), atten_weight.size(1)
            a = a.cpu().numpy()
            for i in range(bsz):
                attn_i = a[i]
                draw(attn_i, src_token[i], src_dict, None, "/mnt/libei/fairseq-0.10.2/checkpoints/iwslt-de2en/multi_scale_double_6/"+str(batch_num)+"sentence-"+str(i)+"layer-"+str(layer_num)+".jpg")
        '''
        '''
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            normal=True,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        '''
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerUpSampleEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, fc1_in_dim, fc1_out_dim, fc2_in_dim, fc2_out_dim):
        super().__init__()
        self.fc1_in_dim = fc1_in_dim
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(fc1_in_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            fc1_in_dim,
            fc1_out_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            fc2_in_dim,
            fc2_out_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(fc1_in_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                self.fc1_in_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
        )
        else:
            return RelativeMultiheadAttention(
                self.fc1_in_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if x.size() == residual.size():
            x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

class NormformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.scale_atten = args.scale_atten
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.ffn_norm = args.ffn_norm
        #self.scale_atten = args.scale_atten
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.atten_layer_norm = LayerNorm(self.embed_dim)
        self.ffn_layer_norm = LayerNorm(args.encoder_ffn_embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            if self.scale_atten:
                return ScaleMultiheadAttention(
                    embed_dim,
                    args.encoder_attention_heads,
                    dropout=args.attention_dropout,
                    self_attention=True,
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                )
            else:
                return MultiheadAttention(
                    embed_dim,
                    args.encoder_attention_heads,
                    dropout=args.attention_dropout,
                    self_attention=True,
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                )
        else:
            return ScaleRelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.atten_layer_norm(x)
        x = self.residual_connection(x, residual)
        
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_norm:
            x = self.ffn_layer_norm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerReAttentionEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
        )
        else:
            return RelativeMultiheadReAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        x = x.cuda()
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerMultiBranchEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.branch_num = args.branch_num
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.layer_norm = LayerNorm(self.embed_dim)
        self.if_skip_ln = args.skip_ln
        #self.layer_norm_ffn = LayerNorm(self.embed_dim)
        self.if_bpe_word = args.bpe_word
        if not self.if_bpe_word:
            self.self_attn = nn.ModuleList([])
            self.self_attn.extend([
                    self.build_self_attention(self.embed_dim, args)
                    for i in range(args.branch_num)
            ])
        else:
            self.sub_word_attn = HighOrderMultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
            self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
            self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
            self.fliter = Linear(self.embed_dim, 1)
            self.self_attn = self.build_self_attention(self.embed_dim, args)
        
        #self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.gate_linear = self.build_fc1(2 * self.embed_dim, 1, self.quant_noise,
                self.quant_noise_block_size,)
        self.gate_linear_ffn = self.build_fc1(2 * self.embed_dim, 1, self.quant_noise,
                self.quant_noise_block_size,)
        self.join_method = args.join_method
        self.ffn_join_method = args.ffn_join_method
        self.if_multi_ffn = args.multi_ffn
        if self.if_multi_ffn:
            self.fc1 = nn.ModuleList([])
            self.fc1.extend([
                self.build_fc1(
                    self.embed_dim,
                    args.encoder_ffn_embed_dim,
                    self.quant_noise,
                    self.quant_noise_block_size,
                )
                for i in range(args.branch_num)
            ])
            self.fc2 = nn.ModuleList([])
            self.fc2.extend([
                self.build_fc2(
                    args.encoder_ffn_embed_dim,
                    self.embed_dim,
                    self.quant_noise,
                    self.quant_noise_block_size,
                )
                for i in range(args.branch_num)
            ])
        else:
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
        )
        else:
            return RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def downsampling(self, sub_word_representation, mapping):
        num_matrix = torch.sum(mapping,1)
        num_matrix = torch.where(num_matrix==0, num_matrix+1, num_matrix)
        return ((mapping.transpose(1,2)@sub_word_representation.transpose(0,1)) / (num_matrix.unsqueeze(2))).transpose(0,1)
    def downsampling1(self, sub_word_representation, mapping):
        
        return ((mapping.transpose(1,2)@sub_word_representation.transpose(0,1))).transpose(0,1)
    def upsampling(self, word_representation, mapping, sub_word_representation):
        """
        mapping (tensor) : shape [batch_size, src_length]
        word_representation : shape [W B C]
        """
        return (mapping.half() @ word_representation.transpose(0,1)).transpose(0,1)
    
    def get_class_token(self, mapping):
        count = torch.sum(mapping, 1).unsqueeze(2).float()  # B L' 1
        class_token = (mapping.float() @ count).squeeze(2)
        ones = class_token - class_token + 1
        class_token = torch.where(class_token>1, 2 * ones, class_token)
        return class_token.long()
    
    def cal_gauss_dist(self, tree, gamma):
        tree_matrix1 = torch.unsqueeze(tree, 2).repeat([1,1,tree.shape[1]])
        tree_matrix2 = tree.unsqueeze(1).repeat([1,tree.shape[1],1])
        dist = torch.abs(tree_matrix2 - tree_matrix1)
        D = 1/((2*math.pi*(gamma**2))**(0.5)) * torch.exp((-0.5/(gamma**2)*dist.pow(2)))
        return D
    def appromixation_mapping(self, tree, gamma):
        D = 1/((2*math.pi*(gamma**2))**(0.5)) * torch.exp((-0.5/(gamma**2)*tree.pow(2)))
        return D


    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, tree, mapping, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        
        if not self.if_bpe_word:
            x_list = []
            for i in range(self.branch_num):
                x_list.append(x)
            result = []
            for i, layers in enumerate(self.self_attn):
                residual = x_list[i]
                if self.normalize_before:
                    if not self.if_skip_ln:
                        y = self.self_attn_layer_norm(x_list[i])
                    else:
                        y = x_list[i]
                y, _ = layers(
                    query=y,
                    key=y,
                    value=y,
                    key_padding_mask=encoder_padding_mask,
                    attn_mask=attn_mask,
                )
                # x = F.dropout(x, p=self.dropout + (step // 1000) * 0.01 if self.dropout + (step // 1000) * 0.01 < 0.3 else 0.3, training=self.training)
                y = self.dropout_module(y)
                if self.join_method != "norm-res":
                    y = self.residual_connection(y, residual)
                if not self.normalize_before:
                    y = self.self_attn_layer_norm(y)
                result.append(y)
        else:
            x_list = []
            for i in range(self.branch_num):
                x_list.append(x)
            result = []

            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_word = self.downsampling(x_list[0], tree)
            # 计算子词级别的自注意力
            x_word = self.word_attn_layer_norm(x_word)
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word  
            
            residual = x_word
            if self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word
            
            residual = x_list[0]
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x_list[0], attn_weights = self.sub_word_attn(
                query=x_list[0],
                key=x_list[0],
                value=x_list[0],
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            
            
            x_list[0] = self.dropout_module(x_list[0])
            x_list[0] = self.residual_connection(x_list[0], residual)
            if not self.normalize_before:
                x_list[0] = self.sub_word_attn_layer_norm(x_list[0])
            result.append(x_list[0])

            residual = x_list[1]
            if self.normalize_before:
                y = self.self_attn_layer_norm(x_list[1])
            y, _ = self.self_attn(
                query=y,
                key=y,
                value=y,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            # x = F.dropout(x, p=self.dropout + (step // 1000) * 0.01 if self.dropout + (step // 1000) * 0.01 < 0.3 else 0.3, training=self.training)
            y = self.dropout_module(y)
            y = self.residual_connection(y, residual)
            if not self.normalize_before:
                y = self.self_attn_layer_norm(y)
            result.append(y)


        
        if self.join_method == "norm-add":
            for i in range(len(result)):
                result[i] = self.layer_norm(result[i])
            x = sum(result)/self.branch_num
        elif self.join_method == "gate":
            alpha = torch.sigmoid(self.gate_linear(torch.cat(result, dim=-1)))
            x = alpha * result[0] + (1 - alpha) * result[1]
        elif self.join_method == "mean":
            x = sum(result)/self.branch_num
        elif self.join_method == "add":
            x = sum(result)
        elif self.join_method == "norm":
            for i in range(len(result)):
                result[i] = self.layer_norm(result[i])
            x = sum(result)
        elif self.join_method == "norm-res":
            for i in range(len(result)):
                result[i] = self.layer_norm(result[i])
            x1 = sum(result)
            x = math.sqrt(0.5)*(x + x1)
        elif self.join_method == "norm-mul":
            for i in range(len(result)):
                result[i] = self.layer_norm(result[i])
            x = math.sqrt(0.5) * sum(result)


        

        if self.if_multi_ffn:
            fc1_result = []
            for fc1, fc2 in zip(self.fc1, self.fc2):
                residual = x
                if self.normalize_before:
                    if not self.if_skip_ln:
                        y = self.final_layer_norm(x)
                    else:
                        y = x
                y = self.activation_fn(fc1(y))
                y = self.activation_dropout_module(y)
                y = fc2(y)
                y = self.dropout_module(y)
                # x = F.dropout(x, p=self.dropout + (step // 1000) * 0.01 if self.dropout + (step // 1000) * 0.01 < 0.3 else 0.3, training=self.training)
                if self.join_method != "norm-res":
                    y = self.residual_connection(y, residual)
                fc1_result.append(y)
            if self.ffn_join_method == "norm-add":
                for i in range(len(fc1_result)):
                    fc1_result[i] = self.layer_norm(fc1_result[i])
                x = sum(fc1_result)/self.branch_num
            elif self.ffn_join_method == "gate":
                alpha = torch.sigmoid(self.gate_linear(torch.cat(fc1_result, dim=-1)))
                x = alpha * fc1_result[0] + (1 - alpha) * fc1_result[1]
            elif self.ffn_join_method == "mean":
                x = sum(fc1_result)/self.branch_num
            elif self.ffn_join_method == "add":
                x = sum(fc1_result)
            elif self.join_method == "norm":
                for i in range(len(fc1_result)):
                    fc1_result[i] = self.layer_norm(fc1_result[i])
                x = sum(fc1_result)
            elif self.join_method == "norm-res":
                for i in range(len(fc1_result)):
                    fc1_result[i] = self.layer_norm(fc1_result[i])
                x1 = sum(fc1_result)
                x = math.sqrt(0.5) * (x + x1)
            elif self.join_method == "norm-mul":
                for i in range(len(fc1_result)):
                    fc1_result[i] = self.layer_norm(fc1_result[i])
                x = math.sqrt(0.5) * sum(fc1_result)


        else:
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)

            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
        return x
    
    def sim_of_branch(self, branch1, branch2):
        branch1, branch2 = branch1.transpose(0,1), branch2.transpose(0,1)
        cos = torch.cosine_similarity(branch1, branch2)
        return torch.sum(torch.sum(cos, dim=-1) / branch1.size(1))/branch1.size(0)
        


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.non_ffn = args.non_ffn
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        if not self.non_ffn:
            self.activation_fn = utils.get_activation_fn(
                activation=str(args.activation_fn)
                if getattr(args, "activation_fn", None) is not None
                else "relu"
            )
            activation_dropout_p = getattr(args, "activation_dropout", 0)

            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = getattr(args, "relu_dropout", 0)
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        if not self.non_ffn:
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.decoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.decoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )

            self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
        )
        else:
            return RelativeMultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        if not self.non_ffn:
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)

            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

class NormformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.scale_atten = args.scale_atten
        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.ffn_norm = args.ffn_norm
        #self.scale_atten = args.scale_atten
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.atten_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.ffn_layer_norm = LayerNorm(args.decoder_ffn_embed_dim, export=export)
        self.cross_atten_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        if args.max_relative_length == -1:
            if self.scale_atten:
                return ScaleMultiheadAttention(
                    embed_dim,
                    args.decoder_attention_heads,
                    dropout=args.attention_dropout,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    self_attention=not getattr(args, "cross_self_attention", False),
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                )
            else:
                return MultiheadAttention(
                    embed_dim,
                    args.decoder_attention_heads,
                    dropout=args.attention_dropout,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    self_attention=not getattr(args, "cross_self_attention", False),
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                )
        else:
            return ScaleRelativeMultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def build_encoder_attention(self, embed_dim, args):
        if self.scale_atten:
            return ScaleMultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.atten_layer_norm(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.cross_atten_layer_norm(x)
            x = self.residual_connection(x, residual)
            
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_norm:
            x = self.ffn_layer_norm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class PATransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.qk_free = args.qk_free
        self.need_attn = True

        self.onnx_trace = False

    def build_self_attention(
            self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        if args.max_relative_length == -1:
            return EfficientPAQKMultiheadAttention(
                args,
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                decoder=True,
            )

    def build_encoder_attention(self, embed_dim, args):
        return EfficientPAQKMultiheadAttention(
            args,
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            decoder=True
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
            self,
            x,
            attn_before: Optional[torch.Tensor] = None,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if self.qk_free:
            if prev_self_attn_state is not None:
                prev_value = prev_self_attn_state[:1]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_value": prev_value,
                }
                if len(prev_self_attn_state) >= 2:
                    saved_state["prev_key_padding_mask"] = prev_self_attn_state[1]
                assert incremental_state is not None
                self.self_attn._set_input_buffer(incremental_state, saved_state)
        else:
            if prev_self_attn_state is not None:
                prev_key, prev_value = prev_self_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_self_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
                assert incremental_state is not None
                self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
            attn_before=attn_before,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def compute_macs_params(self, T=1, S=1):
        macs = 0
        n_params = 0
        macs_attn = 0

        # LayerNorm
        n_params += sum([p.numel() for p in self.self_attn_layer_norm.parameters()])
        # n_params += sum([p.numel() for p in self.final_layer_norm.parameters()])

        # self attention
        self_attn_layer = self.self_attn.compute_macs_params(T=T, S=T)
        macs += self_attn_layer['macs']
        n_params += self_attn_layer['params']
        macs_attn += self_attn_layer['macs_attn']

        # Encoder-decoder attn
        if self.encoder_attn is not None:
            # self attention scaled-dot-product Attn
            enc_attn = self.encoder_attn.compute_macs_params(T=T, S=S)
            macs += enc_attn['macs']
            n_params += enc_attn['params']
            macs_attn += enc_attn['macs_attn']

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': macs_attn
        }


class EITransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
            self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:

            return RelativeMultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

    def build_encoder_attention(self, embed_dim, args):
        return EfficientEnhancedMultiHeadAttention(
            args,
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn



class PADecoderSANGlobalAGenLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_graph_generation(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True

        self.onnx_trace = False


    def build_graph_generation(
            self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        if args.emha:
            return UniversalDynamicGraphStructureGeneration(
                args,
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        elif args.standard_mhsa:
            return UniversalStandardDynamicGraphStructureGeneration(
                args,
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key = prev_self_attn_state[:1]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
            }
            if len(prev_self_attn_state) >= 2:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[1]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask
        )



        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"]]
            return attn, self_attn_state
        return attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m



class TransformerEncoderLayer_layer_level(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.method = args.method
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        if self.method == 1:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim)              
           #self.fliter1 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
           self.fliter2 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
           #args.encoder_embed_dim = 2*self.embed_dim
           #self.inner_block = TransformerEncoderLayer(args)
           #args.encoder_embed_dim = self.embed_dim
           #self.gate = Linear(2*self.embed_dim, 1)
        if self.method == 2:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim)              
           #self.fliter1 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
           #self.fliter2 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
           #args.encoder_embed_dim = 2*self.embed_dim
           #self.inner_block = TransformerEncoderLayer(args)
           #args.encoder_embed_dim = self.embed_dim
           #self.gate = Linear(2*self.embed_dim, 1)
        if self.method == 3:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim)              
           self.fliter1 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
           self.fliter2 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
           #args.encoder_embed_dim = 2*self.embed_dim
           #self.inner_block = TransformerEncoderLayer(args)
           #args.encoder_embed_dim = self.embed_dim
           #self.gate = Linear(2*self.embed_dim, 1)
        if self.method == 4:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim)  
           self.word_attn = self.build_self_attention(2*self.embed_dim, args)   # sub-word attention
           self.word_attn_layer_norm = LayerNorm(2*self.embed_dim)              
           self.fliter1 = Linear(self.embed_dim, 2*self.embed_dim)     # cross attnetion
           self.fliter2 = Linear(2*self.embed_dim, self.embed_dim)     # cross attnetion
           #args.encoder_embed_dim = 2*self.embed_dim
           #self.inner_block = TransformerEncoderLayer(args)
           #args.encoder_embed_dim = self.embed_dim
           #self.gate = Linear(2*self.embed_dim, 1)
        if self.method == 5:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim)               
           self.fliter1 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
           #self.fliter2 = Linear(2*self.embed_dim, self.embed_dim)     # cross attnetion
           #args.encoder_embed_dim = 2*self.embed_dim
           #self.inner_block = TransformerEncoderLayer(args)
           #args.encoder_embed_dim = self.embed_dim
           #self.gate = Linear(2*self.embed_dim, 1)
        if self.method == 6:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)               
           self.fliter1 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
           #self.fliter2 = Linear(2*self.embed_dim, self.embed_dim)     # cross attnetion
           #args.encoder_embed_dim = 2*self.embed_dim
           #self.inner_block = TransformerEncoderLayer(args)
           #args.encoder_embed_dim = self.embed_dim
           #self.gate = Linear(2*self.embed_dim, 1)
        if self.method == 7:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)               
           self.fliter1 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
        if self.method == 8:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           #self.word_attn_layer_norm = LayerNorm(self.embed_dim)               
           #self.fliter1 = Linear(self.embed_dim, self.embed_dim)     # cross attnetion
        if self.method == 9:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate = Linear(2*self.embed_dim,1)
           
        if self.method == 28:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate = Linear(2*self.embed_dim,1)
           self.fliter = Linear(self.embed_dim, 1)
        if self.method == 10:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.cross_attn = MultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate = Linear(2*self.embed_dim,1)
        if self.method == 11:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate = Linear(2*self.embed_dim,1)
        if self.method == 12:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate = Linear(2*self.embed_dim,1)
           self.fliter = Linear(self.embed_dim, self.embed_dim)
        if self.method == 13:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate = Linear(2*self.embed_dim,1)
        if self.method == 14:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate = Linear(2*self.embed_dim,1)
           self.gate_inner = Linear(2*self.embed_dim,1)
        if self.method == 15:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate = Linear(2*self.embed_dim,1)
        if self.method == 16:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           #self.gate = Linear(2*self.embed_dim,1)
        if self.method == 17:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate_word = Linear(2*self.embed_dim,1)
           self.gate_sub = Linear(2*self.embed_dim,1)
           self.gate_final = Linear(2*self.embed_dim,1)
        if self.method == 18:
           self.sub_word_attn = DynamicMaskMultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.gate = Linear(2*self.embed_dim,1)
        if self.method == 20:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
           self.gate = Linear(2*self.embed_dim,1)
        if self.method == 21:
           self.sub_word_attn = self.build_self_attention(self.embed_dim, args)   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)
        if self.method == 22:
           self.sub_word_attn = MsMultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
        if self.method == 23:
           self.sub_word_attn = MsMultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
           self.fliter = SElayer(self.embed_dim)
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
        if self.method == 24:
           self.sub_word_attn = MsMultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
        if self.method == 25:
           self.sub_word_attn = MsMultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
           self.fliter = Linear(self.embed_dim, 1)
        if self.method == 26:
           self.sub_word_attn = MultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
           self.fliter = Linear(self.embed_dim, 1)
        if self.method == 27:
           self.sub_word_attn = MultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
           self.fliter = Linear(self.embed_dim, 1)
        if self.method == 30:
           self.sub_word_attn = MultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
           self.fliter = Linear(self.embed_dim, 1)
        if self.method == 31:
           self.sub_word_attn = HighOrderMultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
           self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
           self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
           self.fliter = Linear(self.embed_dim, 1)
        
        if self.method == 32:
            self.sub_word_attn = HighOrderMultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention   # sub-word attention
            self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim) 
            self.word_attn_layer_norm = LayerNorm(self.embed_dim)   
            self.fliter = Linear(self.embed_dim, 1)        
        
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.final_layer_norm = LayerNorm(self.embed_dim)
        
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )
    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )
    def build_cross_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=False,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
        )
        else:
            return RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads, args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,encoder_decoder_attention=True, self_attention=False, 
            )

    def residual_connection(self, x, residual):
        return residual + x    
    def downsampling(self, sub_word_representation, mapping):
        num_matrix = torch.sum(mapping,1)
        num_matrix = torch.where(num_matrix==0, num_matrix+1, num_matrix)
        return ((mapping.transpose(1,2)@sub_word_representation.transpose(0,1)) / (num_matrix.unsqueeze(2))).transpose(0,1)
    def downsampling1(self, sub_word_representation, mapping):
        
        return ((mapping.transpose(1,2)@sub_word_representation.transpose(0,1))).transpose(0,1)
    def upsampling(self, word_representation, mapping, sub_word_representation):
        """
        mapping (tensor) : shape [batch_size, src_length]
        word_representation : shape [W B C]
        """
        return (mapping.half() @ word_representation.transpose(0,1)).transpose(0,1)
    
    def get_class_token(self, mapping):
        count = torch.sum(mapping, 1).unsqueeze(2).float()  # B L' 1
        class_token = (mapping.float() @ count).squeeze(2)
        ones = class_token - class_token + 1
        class_token = torch.where(class_token>1, 2 * ones, class_token)
        return class_token.long()
    
    def cal_gauss_dist(self, tree, gamma):
        tree_matrix1 = torch.unsqueeze(tree, 2).repeat([1,1,tree.shape[1]])
        tree_matrix2 = tree.unsqueeze(1).repeat([1,tree.shape[1],1])
        dist = torch.abs(tree_matrix2 - tree_matrix1)
        D = 1/((2*math.pi*(gamma**2))**(0.5)) * torch.exp((-0.5/(gamma**2)*dist.pow(2)))
        return D
    def appromixation_mapping(self, tree, gamma):
        D = 1/((2*math.pi*(gamma**2))**(0.5)) * torch.exp((-0.5/(gamma**2)*tree.pow(2)))
        return D
    
    
    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
    
    def forward(self, x, tree, mapping, encoder_padding_mask, embedding, class_embedding, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        if self.method == 1:
            # 通过tree生成词级别的特征   tree shape [batch, src_length]
            # ########################################### optimizer
            #num_words = tree.shape[2]     # 最大词量
            #print(mapping.shape)
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            #x_sub = self.fliter1(x)
            #x_sub = F.relu(x_sub)
            #x_sub = self.dropout_module(x_sub)
            #x_sub = self.fliter2(x_sub)
            #x_sub = self.dropout_module(x_sub)
            x_sub = x
            x_word = self.fliter2(x)
            x_word = self.dropout_module(x_word)
            x_word = self.downsampling(x_word, tree)
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
                
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_sub,
                value=x_sub,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            x = x_sub
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 2:
            # 通过tree生成词级别的特征   tree shape [batch, src_length]
            # ########################################### optimizer
            #num_words = tree.shape[2]     # 最大词量
            #print(mapping.shape)
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            #x_sub = self.fliter1(x)
            #x_sub = F.relu(x_sub)
            #x_sub = self.dropout_module(x_sub)
            #x_sub = self.fliter2(x_sub)
            #x_sub = self.dropout_module(x_sub)
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
                
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_sub,
                value=x_sub,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            x = x_sub
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        
        if self.method == 3:
            # 通过tree生成词级别的特征   tree shape [batch, src_length]
            # ########################################### optimizer
            #num_words = tree.shape[2]     # 最大词量
            #print(mapping.shape)
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            #x_sub = self.fliter1(x)
            #x_sub = F.relu(x_sub)
            #x_sub = self.dropout_module(x_sub)
            #x_sub = self.fliter2(x_sub)
            #x_sub = self.dropout_module(x_sub)
            x_sub = x
            x_word = self.fliter1(x)
            x_word = self.dropout_module(x_word)
            x_word = self.fliter2(x)
            x_word = self.dropout_module(x_word)
            x_word = self.downsampling(x, tree)
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
                
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_sub,
                value=x_sub,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            x = x_sub
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
            
            
        if self.method == 4:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            x_sub = x
            x_word = self.fliter1(x)
            x_word = self.dropout_module(x_word)
            
            x_word = self.downsampling(x_word, tree)
            x_word = self.word_attn_layer_norm(x_word)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            x_word = self.fliter2(x_word)
            x_word = self.dropout_module(x_word)
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_sub,
                value=x_sub,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                #x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            x = x_sub
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 5:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            word_pos_embedding = embedding(encoder_padding_mask_word)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.fliter1(x)
            x_word = self.dropout_module(x_word)
            
            x_word = self.downsampling(x_word, tree)
            x_word = x_word + word_pos_embedding.transpose(0,1)
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_sub,
                value=x_sub,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            x = x_sub
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 6:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            word_pos_embedding = embedding(encoder_padding_mask_word)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.fliter1(x)
            x_word = self.downsampling(x_word, tree)
            x_word = x_word + word_pos_embedding.transpose(0,1)
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_sub,
                value=x_sub,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            attn_weights_super = tree.transpose(1,2) @ attn_weights @ tree
            x_sub2word = (attn_weights_super @ x_word.transpose(0,1)).transpose(0,1)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            x_word = x_word + x_sub2word
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            x = x_sub
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 7:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word_temp = encoder_padding_mask_word + 1
            word_pos_embedding = embedding(encoder_padding_mask_word_temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.fliter1(x)
            x_word = self.downsampling(x_word, tree)
            x_word = x_word + word_pos_embedding.transpose(0,1)
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_sub,
                value=x_sub,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            attn_weights_super = tree.transpose(1,2) @ attn_weights @ tree
            x_sub2word = (attn_weights_super @ x_word.transpose(0,1)).transpose(0,1)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            x_word = x_word + x_sub2word
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            x = x_sub
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 8:
            
            class_token = self.get_class_token(tree.long())
            class_token = class_embedding(class_token)
            x = x + class_token.transpose(0,1)
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 9:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_sub], 2)))
            x = alpha *x + (1-alpha) * x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 10:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.cross_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_sub], 2)))
            x = alpha *x + (1-alpha) * x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 11:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.final_layer_norm(x_sub)
            x_sub = self.activation_fn(self.fc1(x_sub))
            x_sub = self.activation_dropout_module(x_sub)
            x_sub = self.fc2(x_sub)
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.final_layer_norm(x_sub)
            
            
            
            
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_sub], 2)))
            x = alpha *x + (1-alpha) * x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
            
        if self.method == 12:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            x_word = self.fliter(x_word)
            x_word = self.dropout_module(x_word)
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_sub], 2)))
            x = alpha *x + (1-alpha) * x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
            
        if self.method == 13:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_sub], 2)))
            x = alpha *x + (1-alpha) * x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 14:
            # generate world level padding 
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            # downsampling to generate word level representation
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            
            # rightest branch   word level representation and sub-word level representation interaction
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, cross_attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            # leftest branch  sub-word level representation and sub-word level representation interaction
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            x, self_attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
                
                
                
            # left branch and right branch's interaction    
                
            # step1 hyper-graph construction
            attn_weights_super = tree.transpose(0,1) @ self_attn_weights @ tree
            
            # step2 aggregation
            x_sub2word = attn_weights_super @ x_word
            
            # step3 aggregation
            x_sub2word2sub = cross_attn_weights @ x_sub2word
            
            # sum
            alpha_inner = torch.sigmoid(self.gate_inner(torch.cat([x_sub,x_sub2word2sub], 2)))
            x_sub = alpha_inner * x_sub + (1-alpha_inner) * x_sub2word2sub
            
            # 
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_sub], 2)))
            x = alpha *x + (1-alpha) * x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
            
        if self.method == 15:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights_sub = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
                
            # step1 hyper-graph construction
            #print(tree.shape)
            #print(attn_weights.shape)
            attn_weights_super = tree.transpose(1,2) @ attn_weights_sub @ tree
            
            # step2 aggregation
            x_sub2word = (attn_weights_super @ x_word.transpose(0,1)).transpose(0,1)
            
            
            
            
            
            
            
            
            residual = x_word
            #residual_orig = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            
            x_word = 0.5*x_word + 0.5*x_sub2word
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, cross_attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            
            
            
            
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_sub], 2)))
            x = alpha *x + (1-alpha) * x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 16:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, cross_attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            x = x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 17:
            # 生成词level特征的embedding
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            # 定义变量   x_sub  子词level特征     x_word 词level特征
            
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word_after_attn, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word_after_attn = self.dropout_module(x_word_after_attn)
            x_word_after_attn = self.residual_connection(x_word_after_attn, residual)
            if not self.normalize_before:
                x_word_after_attn = self.word_attn_layer_norm(x_word_after_attn)
            
            
            
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            # 计算子词级别的自注意力
            x_sub_after_attn, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_sub,
                value=x_sub,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x_sub_after_attn = self.dropout_module(x_sub_after_attn)
            x_sub_after_attn = self.residual_connection(x_sub_after_attn, residual)
            if not self.normalize_before:
                x_sub_after_attn= self.sub_word_attn_layer_norm(x_sub_after_attn)
            
            
            
            
            
            # word2sub
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word_after_attn = self.word_attn_layer_norm(x_word_after_attn)
            # 计算子词级别的自注意力
            x_word2sub, cross_attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word_after_attn,
                value=x_word_after_attn,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word2sub = self.dropout_module(x_word2sub)
            x_word2sub = self.residual_connection(x_word2sub, residual)
            if not self.normalize_before:
                x_word2sub = self.sub_word_attn_layer_norm(x_word2sub)
            
            
            
            # sub2word
            
            
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
                x_sub_after_attn = self.sub_word_attn_layer_norm(x_sub_after_attn)
            # 计算子词级别的自注意力
            x_sub2word, cross_attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_sub_after_attn,
                value=x_sub_after_attn,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x_sub2word = self.dropout_module(x_sub2word)
            x_sub2word = self.residual_connection(x_sub2word, residual)
            if not self.normalize_before:
                x_sub2word = self.word_attn_layer_norm(x_sub2word)
            
            alpha_word = torch.sigmoid(self.gate_word(torch.cat([x_word_after_attn, x_sub2word],-1)))
            x_word_final = alpha_word * x_word_after_attn + (1-alpha_word) * x_sub2word
            alpha_sub = torch.sigmoid(self.gate_sub(torch.cat([x_sub_after_attn, x_word2sub],-1)))
            x_sub_final = alpha_sub * x_sub_after_attn + (1-alpha_sub) * x_word2sub
            
            
            
            residual = x_sub_final
            if self.normalize_before:
                x_sub_final = self.final_layer_norm(x_sub_final)
            x_sub_final = self.activation_fn(self.fc1(x_sub_final))
            x_sub_final = self.activation_dropout_module(x_sub_final)
            x_sub_final = self.fc2(x_sub_final)
            x_sub_final = self.dropout_module(x_sub_final)
            x_sub_final = self.residual_connection(x_sub_final, residual)
            if not self.normalize_before:
                x_sub_final = self.final_layer_norm(x_sub_final)
                
            residual = x_word_final
            if self.normalize_before:
                x_word_final = self.final_layer_norm(x_word_final)
            x_word_final = self.activation_fn(self.fc1(x_word_final))
            x_word_final = self.activation_dropout_module(x_word_final)
            x_word_final = self.fc2(x_word_final)
            x_word_final = self.dropout_module(x_word_final)
            x_word_final = self.residual_connection(x_word_final, residual)
            if not self.normalize_before:
                x_word_final = self.final_layer_norm(x_word_final)
            
            
            x_word2sub_final = self.upsampling(x_word_final,tree,x_sub_final)
            alpha_final = torch.sigmoid(self.gate_final(torch.cat([x_sub_final, x_word2sub_final],-1)))
            x_final = alpha_final * x_sub_final + (1-alpha_final) * x_word2sub_final
            return x_final, None
        if self.method == 18:
            dist_mask = self.cal_gauss_dist(mapping)
            full_mask = torch.ones_like(dist_mask)
            x_word = x 
            # branch1
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                mask = full_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            
            # branch2
            residual = x_word
            if self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                mask = dist_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
                   
            
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_word], 2)))
            x = alpha *x + (1-alpha) * x_word
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 19:
            dist_mask = self.cal_gauss_dist(mapping)
            full_mask = torch.ones_like(dist_mask)
            x_word = x 
            # branch1
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                mask = full_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            
            # branch2
            residual = x_word
            if self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                mask = dist_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
                   
            
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_word], 2)))
            x = alpha *x + (1-alpha) * x_word
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 20:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            #x_word = self.word_attn_layer_norm(x_word)
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                #x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_sub], 2)))
            x = alpha *x + (1-alpha) * x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
            
        if self.method == 21:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_original = x
            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
                
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
           
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 22:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                key_word=x_word,
                value=x,
                value_word=x_word,
                key_padding_mask=encoder_padding_mask,
                key_padding_mask_word=encoder_padding_mask_word,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 23:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                key_word=x_word,
                value=x,
                value_word=x_word,
                key_padding_mask=encoder_padding_mask,
                key_padding_mask_word=encoder_padding_mask_word,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            x = x.permute([1,2,0])
            x = self.fliter(x)
            x = x.permute([2,0,1])
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 24:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            tree = 1-tree
            tree = self.appromixation_mapping(tree, 0.5)
            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.sub_word_attn_layer_norm(x_word)
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                key_word=x_word,
                value=x,
                value_word=x_word,
                key_padding_mask=encoder_padding_mask,
                key_padding_mask_word=encoder_padding_mask_word,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
            
        if self.method == 25:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.word_attn_layer_norm(x_word)
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                key_word=x_word,
                value=x,
                value_word=x_word,
                key_padding_mask=encoder_padding_mask,
                key_padding_mask_word=encoder_padding_mask_word,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 26:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
                
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
                
            x_word2sub = self.upsampling(x_word, tree, x)
            x = 0.5 * x + 0.5 * x_word2sub
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        
        if self.method == 27:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.word_attn_layer_norm(x_word)
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word
            x_orig = x
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
                            
         
            residual = x_orig
            if self.normalize_before:
                x_orig = self.sub_word_attn_layer_norm(x_orig)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word2sub, attn_weights = self.sub_word_attn(
                query=x_orig,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
                
            )
            x_word2sub = self.dropout_module(x_word2sub)
            x_word2sub = self.residual_connection(x_word2sub, residual)
            if not self.normalize_before:
                x_word2sub = self.word_attn_layer_norm(x_word2sub)
                
            alpha = torch.sigmoid(self.gate_linear(torch.cat([x, x_word2sub], -1)))
            x = alpha * x + (1-alpha)*x_word2sub
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
            
        if self.method == 28:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_sub = x
            x_word = self.downsampling(x, tree)
            x_word = self.word_attn_layer_norm(x_word)
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word
            residual = x_word
            if self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.word_attn_layer_norm(x_word)
            
            residual = x_sub
            if self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
                x_word = self.word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_sub, attn_weights = self.sub_word_attn(
                query=x_sub,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_sub = self.dropout_module(x_sub)
            x_sub = self.residual_connection(x_sub, residual)
            if not self.normalize_before:
                x_sub = self.sub_word_attn_layer_norm(x_sub)
            
            
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            alpha = torch.sigmoid(self.gate(torch.cat([x,x_sub], 2)))
            x = alpha *x + (1-alpha) * x_sub
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        
        if self.method == 29:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.word_attn_layer_norm(x_word)
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                key_word=x_word,
                value=x,
                value_word=x_word,
                key_padding_mask=encoder_padding_mask,
                key_padding_mask_word=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 30:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.word_attn_layer_norm(x_word)
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word
            
            
            
            
            
            
            
            
            
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word2sub, attn_weights = self.sub_word_attn(
                query=x,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            
            x = x + x_word2sub
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
                
            
            
            
                
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
        if self.method == 31:
            max_words = torch.max(mapping) -4
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -4)).unsqueeze(1) + 1
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp < max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
            
            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.word_attn_layer_norm(x_word)
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word  
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x_word2sub, attn_weights = self.sub_word_attn(
                query=x,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            
            x = x + x_word2sub
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
                
            
            
            
                
            
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None
            
        if self.method == 32:
            max_words = torch.max(mapping) -3
            max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -3)).unsqueeze(1)
            temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
            zero = temp - temp
            encoder_padding_mask_word = torch.where(temp <= max_words_per_sentence, zero, temp)
            encoder_padding_mask_word = encoder_padding_mask_word.eq(0)

            x_word = self.downsampling(x, tree)
            # 计算子词级别的自注意力
            x_word = self.word_attn_layer_norm(x_word)
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word  
            #print(torch.sigmoid(self.fliter(x_word)))
            residual = x_word
            if self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            # 计算子词级别的自注意力
            x_word, attn_weights = self.sub_word_attn(
                query=x_word,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=attn_mask,
            )
            x_word = self.dropout_module(x_word)
            x_word = self.residual_connection(x_word, residual)
            if not self.normalize_before:
                x_word = self.sub_word_attn_layer_norm(x_word)
            
            x_word = torch.sigmoid(self.fliter(x_word)) * x_word
            #print(torch.sigmoid(self.fliter(x_word)))
            residual = x
            if self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
            # 计算子词级别的自注意力
            x_word2sub, attn_weights = self.sub_word_attn(
                query=x,
                key=x_word,
                value=x_word,
                key_padding_mask=encoder_padding_mask_word,
                attn_mask=None,
            )
            x, attn_weights = self.sub_word_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x = x + x_word2sub
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.sub_word_attn_layer_norm(x)
               
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x, None

class TransformerEncoderLayer_layer_level_clean(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.method = args.method
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.sub_word_attn = MultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )   # sub-word attention
        self.sub_word_attn_layer_norm = LayerNorm(self.embed_dim)
        self.join_layer_norm = LayerNorm(self.embed_dim)
        self.gate = Linear(2*self.embed_dim, 1)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.final_layer_norm = LayerNorm(self.embed_dim)
        
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )
    def build_self_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )
    def build_cross_attention(self, embed_dim, args):
        if args.max_relative_length == -1:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=False,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
        )
        else:
            return RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads, args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,encoder_decoder_attention=True, self_attention=False, 
            )

    def residual_connection(self, x, residual):
        return residual + x    
    def downsampling(self, sub_word_representation, mapping):
        num_matrix = torch.sum(mapping,1)
        num_matrix = torch.where(num_matrix==0, num_matrix+1, num_matrix)
        return ((mapping.transpose(1,2)@sub_word_representation.transpose(0,1)) / torch.sqrt((num_matrix.unsqueeze(2)))).transpose(0,1)
    def downsampling1(self, sub_word_representation, mapping):
        
        return ((mapping.transpose(1,2)@sub_word_representation.transpose(0,1))).transpose(0,1)
    def upsampling(self, word_representation, mapping, sub_word_representation):
        """
        mapping (tensor) : shape [batch_size, src_length]
        word_representation : shape [W B C]
        """
        return (mapping.half() @ word_representation.transpose(0,1)).transpose(0,1)
    
    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
    
    def forward(self, x, tree, mapping, encoder_padding_mask, embedding, class_embedding, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        max_words = torch.max(mapping) -3
        max_words_per_sentence = (max_words - (torch.max(mapping, 1)[0] -3)).unsqueeze(1)
        temp = torch.arange(1, max_words+1).unsqueeze(0).repeat([x.shape[1], 1]).to(mapping.device)
        zero = temp - temp
        encoder_padding_mask_word = torch.where(temp <= max_words_per_sentence, zero, temp)
        encoder_padding_mask_word = encoder_padding_mask_word.eq(0)
        # generate word representation
        
        x_word = self.downsampling(x, tree)
        x_word = self.sub_word_attn_layer_norm(x_word)
        #scale = x.shape[-1] ** (-0.5)
        #x_word = torch.sigmoid(self.fliter(x_word * scale)) * x_word
        residual = x
        if self.normalize_before:
            x = self.sub_word_attn_layer_norm(x)
        # 计算子词级别的自注意力
        x, attn_weights = self.sub_word_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.sub_word_attn_layer_norm(x)

        residual = x_word
        if self.normalize_before:
            x_word = self.sub_word_attn_layer_norm(x_word)
        # 计算子词级别的自注意力
        x_word, attn_weights = self.sub_word_attn(
            query=x_word,
            key=x_word,
            value=x_word,
            key_padding_mask=encoder_padding_mask_word,
            attn_mask=attn_mask,
        )
        x_word = self.dropout_module(x_word)
        x_word = self.residual_connection(x_word, residual)
        if not self.normalize_before:
            x_word = self.sub_word_attn_layer_norm(x_word)
        
        x_word2sub = self.sub_word_attn_layer_norm(self.upsampling(x_word, tree, x))
        #x_word2sub = self.upsampling(x_word, tree, x)
        #x = self.join_layer_norm(x)
        #x_word2sub = self.join_layer_norm(x_word2sub)
        alpha = torch.sigmoid(self.gate(torch.cat([x,x_word2sub],-1)))
        x = alpha * x + (1-alpha) * x_word2sub
        print(1-alpha)
        #x = 0.8 * x + 0.2 * x_word2sub
        
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, None
