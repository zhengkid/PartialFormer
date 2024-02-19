# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
from .dynamic_convolution import DynamicConv1dTBC

bmm_fp16_support = tuple(int(x) for x in torch.version.cuda.split('.')) >= (9, 1, 0)


@with_incremental_state
class MultiheadAttentionFFN(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            args,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.non_init = args.non_init
        self.num_heads = num_heads
        self.add_ln = args.add_ln
        
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.add_res = args.add_res
        self.non_ffn = args.non_ffn
        self.not_sharing = args.not_sharing
        
        if self.add_ln:
            self.q_ln = nn.LayerNorm(embed_dim)
            self.k_ln = nn.LayerNorm(embed_dim)
            self.ffn_ln = nn.LayerNorm(self.head_dim)
        
        if not self.non_ffn:
            if self.not_sharing:
                self.ff1 = quant_noise(
                    nn.Conv1d(self.embed_dim, args.decoder_hidden_ratio * self.embed_dim, stride=1,
                              kernel_size=1, padding=0,
                              groups=self.num_heads), q_noise, qn_block_size
                )
                self.ff2 = quant_noise(nn.Conv1d(args.decoder_hidden_ratio * self.embed_dim, self.embed_dim, stride=1,
                                                 kernel_size=1, padding=0,
                                                 groups=self.num_heads), q_noise,
                                       qn_block_size
                                       )
            else:
                self.ff1 = quant_noise(nn.Linear(self.head_dim, args.decoder_hidden_ratio * self.head_dim, bias=bias), q_noise, qn_block_size)
                self.ff2 = quant_noise(nn.Linear(args.decoder_hidden_ratio * self.head_dim, self.head_dim, bias=bias), q_noise, qn_block_size)
        self.non_gate = args.non_gate
        self.enhance_glu = args.enhanced_glu
        if not args.non_gate:
            if self.enhance_glu:
                self.u_proj1 = quant_noise(
                    nn.Linear(self.head_dim, self.head_dim, bias=bias), q_noise, qn_block_size
                )
                self.u_proj = quant_noise(
                    nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
                )
            else:
                self.u_proj = quant_noise(
                    nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
                )

        self.head_interact = args.head_interact
        if self.head_interact:
            self.head_ff1 = quant_noise(
                nn.Linear(self.num_heads, args.decoder_head_hidden_ratio * self.num_heads, bias=bias), q_noise, qn_block_size
            )
            self.head_ff2 = quant_noise(nn.Linear(args.decoder_head_hidden_ratio * self.num_heads, self.num_heads, bias=bias),
                                        q_noise,
                                        qn_block_size
                                        )
        self.gate_before = args.gate_before

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

        self.add_dropout = args.add_dropout
        if self.add_dropout:
            self.ffn_dropout_module = FairseqDropout(
                args.hidden_dropout, module_name=self.__class__.__name__
            )



        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            if not self.non_gate:
                if not self.non_init:
                    if self.enhance_glu:
                        nn.init.xavier_uniform_(self.u_proj.weight, gain=1 / math.sqrt(2))
                        nn.init.xavier_uniform_(self.u_proj1.weight, gain=1 / math.sqrt(2))
                    else:
                        nn.init.xavier_uniform_(self.u_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
            if not self.non_gate:
                if not self.non_init:
                    if self.enhance_glu:
                        nn.init.xavier_uniform_(self.u_proj.weight)
                        nn.init.xavier_uniform_(self.u_proj1.weight)
                    else:
                        nn.init.xavier_uniform_(self.u_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            attn_before: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        """ 
        if (
            not self.onnx_trace
            and not self.tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None

            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
            """
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        if not self.non_gate:
            u = self.u_proj(query)
        else:
            u = None
        if self.add_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )

        if u is not None:
            if self.gate_before:
                u = (
                    u.contiguous()
                        .view(-1, bsz * self.num_heads, self.head_dim)
                        .transpose(0, 1)
                )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttentionFFN._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_before is not  None:
            attn_weights = attn_weights + attn_before
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        # res = attn_weights.view(bsz,self.num_heads, tgt_len, src_len)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        # res = attn_weights
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)

        if not self.non_gate:
            if self.gate_before:
                attn = attn * F.relu(u)

        if self.head_interact:
            attn = attn.view(bsz, self.num_heads, tgt_len, self.head_dim).permute(0, 2, 3, 1)
            if self.add_res:
                res = attn
            attn = self.activation_dropout_module(self.activation_fn(self.head_ff1(attn)))
            attn = self.head_ff2(attn)
            if self.add_res:
                attn = attn + res
            attn = attn.permute(0, 3, 1, 2).contiguous().view(-1, tgt_len, self.head_dim)
        

        if not  self.non_ffn:
            if not self.not_sharing:
                if self.add_res:
                    res = attn
                if self.add_ln:
                    attn = self.ffn_ln(attn)
                if self.enhance_glu:
                    g1 = self.u_proj1(res)
                attn = self.activation_dropout_module(self.activation_fn(self.ff1(attn)))
                attn = self.ff2(attn)
                if self.add_dropout:
                    attn = self.ffn_dropout_module(attn)
                if self.enhance_glu:
                    attn = attn * F.relu(g1)

                if self.add_res:
                    attn = attn + res






        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if not self.non_ffn:
            if self.not_sharing:
                if self.add_res:
                    res = attn
                if self.add_ln:
                    attn = self.ffn_ln(attn)
                if self.enhance_glu:
                    g1 = self.u_proj1(res)
                attn = attn.permute(1, 2, 0)
                attn = self.activation_dropout_module(self.activation_fn(self.ff1(attn)))
                attn = self.ff2(attn)
                if self.add_dropout:
                    attn = self.ffn_dropout_module(attn)
                if self.enhance_glu:
                    attn = attn * F.relu(g1)
                attn = attn.permute([2, 0, 1]).contiguous()
                if self.add_res:
                    attn = attn + res

        if not self.non_gate:
            if not self.gate_before:
                attn = attn * F.relu(u)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if True and self.self_attention:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        # if self.self_attention:
        #    print(self.inner_cos_of_attn_weight(attn_weights))
        # res = res.view(bsz, self.num_heads, tgt_len, src_len)
        return attn, attn_weights

    def inner_cos_of_attn_weight(self, atten_weight):
        atten_weights = atten_weight[0]
        result = []
        for i in range(atten_weights.size(0)):
            for j in range(atten_weights.size(0)):
                if i != j:
                    result.append(torch.cosine_similarity(atten_weights[i], atten_weights[j], dim=-1))
        return sum(result)

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                            0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                                                           dim: 2 * dim
                                                           ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    def compute_macs_params(self, T=1, S=1):
        macs = 0
        n_params = 0

        C = self.embed_dim

        # Scaled-dot-product macs
        # [B x T x C] x [B x C x S] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        num_macs_kq = T * S * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = T * C * S

        macs += num_macs_kq + num_macs_v

        if self.self_attention:
            assert T == S
            q_params = sum([p.numel() for p in self.q_proj.parameters()])
            k_params = sum([p.numel() for p in self.k_proj.parameters()])
            v_params = sum([p.numel() for p in self.v_proj.parameters()])
            u_params = sum([p.numel() for p in self.u_proj.parameters()])

            ff1_params = sum([p.numel() for p in self.ff1.parameters()])
            ff2_params = sum([p.numel() for p in self.ff2.parameters()])

            # multiply by Seq length
            macs += (q_params * T) + (k_params * T) + (v_params * T) + (u_params * T) + (self.num_heads * ff1_params * T) + + (self.num_heads * ff2_params * T)
            n_params += q_params + v_params + k_params + u_params + ff1_params + ff2_params
        elif self.encoder_decoder_attention:
            q_params = sum([p.numel() for p in self.q_proj.parameters()])
            k_params = sum([p.numel() for p in self.k_proj.parameters()])
            v_params = sum([p.numel() for p in self.v_proj.parameters()])
            u_params = sum([p.numel() for p in self.u_proj.parameters()])
            ff1_params = sum([p.numel() for p in self.ff1.parameters()])
            ff2_params = sum([p.numel() for p in self.ff2.parameters()])
            print(1)
            macs += (q_params * T) + (k_params * S) + (v_params * S)  + (u_params * T) + (self.num_heads * ff1_params * T) + + (self.num_heads * ff2_params * T)
            n_params += q_params + v_params + k_params + u_params + ff1_params + ff2_params
        else:
            raise NotImplementedError

        out_params = sum([p.numel() for p in self.out_proj.parameters()])
        macs += (out_params * T)
        n_params += out_params

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': num_macs_kq + num_macs_v
        }
