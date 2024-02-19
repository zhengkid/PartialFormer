# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    MultiheadAttention,
    RelativeMultiheadAttention,
    UniversalLightWeightDynamicGraphStructureGeneration,
    UniversalStandardDynamicGraphStructureGeneration,
    UniversalDynamicGraphStructureGeneration,
    UniversalRelativeDynamicGraphStructureGeneration,
    PADecoderSANGlobalAGenLayer,
    UniversalEfficientPaAttentionFFN,
    UniversalEfficientPaAttention,
    SinusoidalPositionalEmbedding,
)
import os
from fairseq.modules.pa_transformer_layer import PATransformerDecoderLayer, PATransformerEncoderLayer
from fairseq.modules.transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
import numpy as np

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("PartialFormer")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--hidden-dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on

        # for RPR
        parser.add_argument('--max-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--k-only', default=False, action='store_true',
                            help='select the relative mode to map relative position information')


        parser.add_argument('--hidden-ratio', type=int, default=4,
                            help='encoder FFN hidden')

        parser.add_argument('--decoder-hidden-ratio', type=int, default=2,
                            help='decoder FFN hidden')

        parser.add_argument('--non-global', default=False, action='store_true',
                            help='wo encoder AG')

        parser.add_argument('--decoder-global', default=False, action='store_true',
                            help='with decoder AG')

        parser.add_argument('--standard-mhsa', default=False, action='store_true',
                            help='A_G choice')


        parser.add_argument('--expand-heads', type=int, default=8,
                            help='number of the  heads')
        parser.add_argument('--global-heads', type=int, default=8,
                            help='number of the  heads')
        parser.add_argument('--decoder-global-heads', type=int, default=8,
                            help='number of the  heads')

        parser.add_argument('--decoder-expand-heads', type=int, default=8,
                            help='number of the  heads')



        parser.add_argument('--add-dropout', default=False, action='store_true',
                            help='FFN dropout')

        parser.add_argument('--use-hdp', default=False, action='store_true',
                            help='complex head scaling')






        # Print  stats
        parser.add_argument('--print-stats', action='store_true', help='Print MACs')
        parser.add_argument('--src-len-ps', type=int, help='Source length for printing stats')
        parser.add_argument('--tgt-len-ps', type=int, help='Target length for printing stats')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if args.print_stats:
            cls.comptue_stats(args, encoder, decoder)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    @classmethod
    def comptue_stats(cls, args, encoder, decoder):

        target_length = args.tgt_len_ps
        source_length = args.src_len_ps

        format_str = "{:<20} | \t {:<10} | \t {:<10}"
        print('=' * 15 * source_length)
        print('{:<90} {:<20}'.format('', cls.__name__))
        print('=' * 15 * source_length)
        print(format_str.format('Layer', 'Params', 'MACs'))
        print('-' * 15 * source_length)
        overall_macs = 0.0
        overall_params = 0.0
        round_places = 2

        # encoder
        import csv
        with open('{}/encoder_stats_t_{}_s_{}.csv'.format(args.save_dir, target_length, source_length),
                  mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for enc_idx, (k, v) in enumerate(encoder.compute_macs_params(src_len=source_length).items()):
                macs = v['macs'] + v['emb_macs']
                params = v['params'] + v['emb_params']

                overall_macs += macs
                overall_params += params

                macs = round(float(macs) / 1e6, round_places)
                params = round(float(params) / 1e6, round_places)

                print(format_str.format(k, params, macs))

                if enc_idx == 0:
                    key_list = list(v.keys())
                    csv_writer.writerow(['Layer'] + key_list)

                value_list = list(v.values())
                value_list = [k] + value_list
                csv_writer.writerow(value_list)

        # print('-' * 60)
        # decoder

        dec_string = {}
        with open('{}/decoder_stats_t_{}_s_{}.csv'.format(args.save_dir, target_length, source_length),
                  mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for time_step in range(1, target_length + 1):
                for dec_idx, (k, v) in enumerate(
                        decoder.compute_macs_params(src_len=source_length, tgt_len=time_step).items()):
                    if args.share_all_embeddings and k == 'Dec_LUT':  # Look-up Table is shared
                        v['emb_params'] = 0

                    macs = v['macs'] + v['emb_macs']
                    params = v['params'] + v['emb_params']

                    overall_macs += macs
                    if time_step == 1:
                        overall_params += params

                    macs = round(float(macs) / 1e6, round_places)
                    params = round(float(params) / 1e6, round_places)

                    if k not in dec_string:
                        dec_string[k] = [[time_step, params, macs]]
                    else:
                        dec_string[k].append([time_step, params, macs])

                    if dec_idx == 0:
                        key_list = list(v.keys())
                        csv_writer.writerow(['Time'] + ['Layer'] + key_list)

                    value_list = list(v.values())
                    value_list = [time_step] + [k] + value_list
                    csv_writer.writerow(value_list)

        format_str_dec1 = '{:<20} | \t '.format("Layer")
        dotted_line = '-' * 20
        for t in range(target_length + 1):
            if t == 0:
                format_str_dec1 += '{:<10} | \t '.format("Params")
            else:
                format_str_dec1 += '{:<10} '.format("t_{}".format(t))
            dotted_line += '-' * 10
        dotted_line += '-' * 10
        format_str_dec1 += '| \t {:<10} '.format("Overall MAC")
        dotted_line += '-' * 10
        # print(format_str_dec)
        print(dotted_line)
        print(format_str_dec1)
        print(dotted_line)

        for layer_name, v in dec_string.items():
            time_step_str = '{:<20} | \t '.format(layer_name)
            macs = 0
            for idx, (t, p, m) in enumerate(v):
                # print(t)
                if idx == 0:
                    time_step_str += '{:<10} | \t '.format(p)
                    time_step_str += '{:<10} '.format(m)
                else:
                    time_step_str += '{:<10} '.format(m)
                macs += m
            time_step_str += '| \t {:<10} '.format(round(macs, round_places))
            print(time_step_str)
        overall_macs = round(float(overall_macs) / 1e6, round_places)
        overall_params = round(float(overall_params) / 1e6, round_places)
        print('-' * 15 * target_length)

        print('Total MACS for {} decoder timesteps: {} M'.format(target_length, overall_macs))
        print('Total parameters: {} M'.format(overall_params))
        print('=' * 15 * target_length)

        with open('{}/overall_stats_t_{}_s_{}.csv'.format(args.save_dir, target_length, source_length),
                  mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Time steps', target_length])
            csv_writer.writerow(['Total MACs (in million)', overall_macs])
            csv_writer.writerow(['Total parameters (in million)', overall_params])

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        
        self.encoder_layerdrop = args.encoder_layerdrop
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        self.non_global = args.non_global
        if not self.non_global:
            self.globa_structure_learning = PATransformerEncoderGen(args)

        # self.sample_idx = 0

    def build_encoder_layer(self, args):
        return PATransformerEncoderLayer(args)


    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not self.non_global:
            graph_structure = self.globa_structure_learning(x, encoder_padding_mask)
        else:
            graph_structure = None

        encoder_states = [] if return_all_hiddens else None
        # encoder layers
        for layer in self.layers:

            x, attn_maps = layer(x, encoder_padding_mask, attn_before=graph_structure)

            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def compute_macs_params(self, src_len=1):
        encoder_mac_params = dict()
        from fairseq.modules.adaptive_input import AdaptiveInput
        if isinstance(self.embed_tokens, nn.Embedding):
            params_emb = self.embed_tokens.weight.numel()

            encoder_mac_params['Enc_LUT'] = {
                'macs': 0,
                'params': 0,
                'macs_attn': 0,
                'emb_macs': 0,  # LUT does not have any MACs
                'emb_params': params_emb
            }
        elif isinstance(self.embed_tokens, AdaptiveInput):
            mac_params_adapt_in = self.embed_tokens.compute_macs_params()
            encoder_mac_params['Enc_Adp_LUT'] = {
                'macs': mac_params_adapt_in['proj_macs'] * src_len,
                'params': mac_params_adapt_in['proj_params'],
                'macs_attn': 0,
                'emb_macs': mac_params_adapt_in['embedding_macs'],
                'emb_params': mac_params_adapt_in['embedding_params']
            }
        else:
            raise NotImplementedError
        if not self.non_global:
            enc_macs_params = self.globa_structure_learning.compute_macs_params(S=src_len)
            encoder_mac_params['Enc_Layer_24'] = {
                'macs': enc_macs_params['macs'],
                'params': enc_macs_params['params'],
                'macs_attn': enc_macs_params['macs_attn'],
                'emb_macs': 0,
                'emb_params': 0
            }
        print(enc_macs_params['macs'])
        for layer_idx, layer in enumerate(self.layers):
            enc_macs_params = layer.compute_macs_params(S=src_len)
            layer_name = 'Enc'
            encoder_mac_params['{}_Layer_{}'.format(layer_name, layer_idx)] = {
                'macs': enc_macs_params['macs'],
                'params': enc_macs_params['params'],
                'macs_attn': enc_macs_params['macs_attn'],
                'emb_macs': 0,
                'emb_params': 0
            }

        if self.layer_norm is not None:
            encoder_mac_params['Enc_LN'] = {
                'macs_attn': 0,
                'macs': 0,
                'params': sum([p.numel() for p in self.layer_norm.parameters()]),
                'emb_macs': 0,
                'emb_params': 0
            }

        return encoder_mac_params


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )
        self.decoder_global = args.decoder_global
        if self.decoder_global:
            self.global_learning = PADecoderSANGlobalAGenLayer(args, no_encoder_attn)

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return PATransformerDecoderLayer(args, no_encoder_attn)


    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            if self.decoder_global:
                if idx == 0:
                    global_attn, _ = self.global_learning(
                        x,
                        encoder_out.encoder_out if encoder_out is not None else None,
                        encoder_out.encoder_padding_mask if encoder_out is not None else None,
                        incremental_state,
                        self_attn_mask=self_attn_mask,
                        self_attn_padding_mask=self_attn_padding_mask,
                        need_attn=bool((idx == alignment_layer)),
                        need_head_weights=bool((idx == alignment_layer)),
                    )
            else:
                global_attn = None

            x, layer_attn, _ = layer(
                x,
                global_attn,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

    def compute_macs_params(self, src_len=1, tgt_len=1):
        decoder_mac_params = dict()
        from fairseq.modules.adaptive_input import AdaptiveInput

        if isinstance(self.embed_tokens, nn.Embedding):
            params_emb = self.embed_tokens.weight.numel()
            decoder_mac_params['Dec_LUT'] = {
                'macs': 0,
                'params': 0,
                'macs_attn': 0,
                'emb_macs': 0,  # LUT does not have any MACs
                'emb_params': params_emb
            }
        elif isinstance(self.embed_tokens, AdaptiveInput):
            mac_params_adapt_in = self.embed_tokens.compute_macs_params()
            decoder_mac_params['Dec_Adp_LUT'] = {
                'macs': mac_params_adapt_in['proj_macs'] * src_len,
                'params': mac_params_adapt_in['proj_params'],
                'macs_attn': 0,
                'emb_macs': mac_params_adapt_in['embedding_macs'],
                'emb_params': mac_params_adapt_in['embedding_params']
            }
        else:
            raise NotImplementedError

        if self.layernorm_embedding is not None:
            decoder_mac_params['Dec_LN_Emb'] = {
                'macs_attn': 0,
                'macs': 0,
                'params': sum([p.numel() for p in self.layernorm_embedding.parameters()]),
                'emb_macs': 0,
                'emb_params': 0
            }
        if self.decoder_global:
            dec_macs_params = self.global_learning.compute_macs_params(S=src_len, T=tgt_len)
            layer_name = 'Dec'
            decoder_mac_params['{}_Layer_{}'.format(layer_name, 7)] = {
                'macs': dec_macs_params['macs'],
                'params': dec_macs_params['params'],
                'macs_attn': dec_macs_params['macs_attn'],
                'emb_macs': 0,
                'emb_params': 0
            }
        for layer_idx, layer in enumerate(self.layers):
            dec_macs_params = layer.compute_macs_params(S=src_len, T=tgt_len)
            layer_name = 'Dec'
            decoder_mac_params['{}_Layer_{}'.format(layer_name, layer_idx)] = {
                'macs': dec_macs_params['macs'],
                'params': dec_macs_params['params'],
                'macs_attn': dec_macs_params['macs_attn'],
                'emb_macs': 0,
                'emb_params': 0
            }

        if self.layer_norm is not None:
            decoder_mac_params['Dec_LN'] = {
                'macs_attn': 0,
                'macs': 0,
                'params': sum([p.numel() for p in self.layer_norm.parameters()]),
                'emb_macs': 0,
                'emb_params': 0
            }

        if self.project_out_dim is not None:
            params_proj = sum([p.numel() for p in self.project_out_dim.parameters()])
            decoder_mac_params['Dec_Proj'] = {
                'macs': params_proj * tgt_len,
                'params': params_proj,
                'macs_attn': 0,
                'emb_macs': 0,
                'emb_params': 0
            }

        # Ouptut layer
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            params_lin = self.embed_tokens.weight.numel()
            if self.share_input_output_embed:
                decoder_mac_params['Dec_Out'] = {
                    'macs': params_lin * tgt_len,
                    'params': 0,
                    'macs_attn': 0,
                    'emb_macs': 0,
                    'emb_params': 0
                }
            else:
                decoder_mac_params['Dec_Out'] = {
                    'macs': params_lin * tgt_len,
                    'params': params_lin,
                    'macs_attn': 0,
                    'emb_macs': 0,
                    'emb_params': 0
                }
        else:
            macs_params_adapt_sm = self.adaptive_softmax.compute_macs_params()
            decoder_mac_params['Dec_Adp_Out'] = {
                'macs': macs_params_adapt_sm['macs'] * tgt_len,
                'params': macs_params_adapt_sm['params'],
                'macs_attn': 0,
                'emb_macs': 0,
                'emb_params': 0
            }

        return decoder_mac_params


class PATransformerEncoderGen(nn.Module):
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
        self.self_attn = self.build_graph_structure(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.normalize_before = args.encoder_normalize_before

    def build_graph_structure(self, embed_dim, args):
        if args.standard_mhsa:
            return UniversalStandardDynamicGraphStructureGeneration(
                args,
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        elif args.rpr:
            return UniversalRelativeDynamicGraphStructureGeneration(args,
                                                                    self.embed_dim, args.encoder_attention_heads,
                                                                    args.max_relative_length,
                                                                    dropout=args.attention_dropout, k_only=args.k_only,
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

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        atten_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask
        )
        return atten_weights

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


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("PartialFormer", "PartialFormer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.hidden_dropout = getattr(args, "hidden_dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)

    args.max_relative_length = getattr(args, 'max_relative_length', args.max_relative_length)
    args.k_only = getattr(args, 'k_only', args.k_only)

    args.hidden_ratio = getattr(args, 'hidden_ratio', args.hidden_ratio)
    args.decoder_hidden_ratio = getattr(args, 'decoder_hidden_ratio', args.decoder_hidden_ratio)



    args.non_global = getattr(args, 'non_global', args.non_global)
    args.decoder_global = getattr(args, 'decoder_global', args.decoder_global)

    args.standard_mhsa = getattr(args, 'standard_mhsa', args.standard_mhsa)

    args.expand_heads = getattr(args, 'expand_heads', args.encoder_attention_heads)

    args.use_hdp = getattr(args, 'use_hdp', args.use_hdp)
    args.add_dropout = getattr(args, 'add_dropout', args.add_dropout)
    args.decoder_expand_heads = getattr(args, 'decoder_expand_heads', args.expand_heads)
    args.global_heads = getattr(args, 'global_heads', args.global_heads)
    args.decoder_global_heads = getattr(args, 'decoder_global_heads', args.decoder_global_heads)

    # Print  stats
    args.print_stats = getattr(args, "print_stats", False)
    args.src_len_ps = getattr(args, "src_len_ps", 20)
    args.tgt_len_ps = getattr(args, "tgt_len_ps", 20)


@register_model_architecture("PartialFormer", "PartialFormer_iwslt_de_en")
def PartialFormer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("PartialFormer", "PartialFormer_t2t_iwslt_de_en")
def PartialFormer_t2t_iwslt_de_en(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    PartialFormer_iwslt_de_en(args)


@register_model_architecture("PartialFormer", "relative_PartialFormer_t2t_iwslt_de_en")
def relative_PartialFormer_t2t_iwslt_de_en(args):
    args.max_relative_length = 8
    args.k_only = True
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    PartialFormer_t2t_iwslt_de_en(args)


@register_model_architecture("PartialFormer", "PartialFormer_wmt_en_de")
def PartialFormer_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("PartialFormer", "PartialFormer_wmt_en_de_big")
def PartialFormer_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture('PartialFormer', 'PartialFormer_t2t_wmt_en_de')
def PartialFormer_t2t_wmt_en_de(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    base_architecture(args)


@register_model_architecture('PartialFormer', 'rpr_PartialFormer_t2t_wmt_en_de')
def rpr_PartialFormer_t2t_wmt_en_de(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.max_relative_length = 8
    args.k_only = True
    base_architecture(args)


@register_model_architecture('PartialFormer', 'PartialFormer_t2t_wmt_en_de_6l')
def PartialFormer_t2t_wmt_en_de_6l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    base_architecture(args)


@register_model_architecture('PartialFormer', 'relative_PartialFormer_wmt_en_de')
def relative_PartialFormer_wmt_en_de(args):
    args.max_relative_length = 8
    args.k_only = True
    base_architecture(args)


@register_model_architecture('PartialFormer', 'relative_PartialFormer_t2t_wmt_en_de')
def relative_PartialFormer_t2t_wmt_en_de(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.max_relative_length = 8
    args.k_only = True
    base_architecture(args)


@register_model_architecture("PartialFormer", "relative_PartialFormer_t2t_wmt_en_de_big")
def relative_PartialFormer_t2t_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    relative_PartialFormer_t2t_wmt_en_de(args)


@register_model_architecture("PartialFormer", "PartialFormer_t2t_wmt_en_de_big")
def PartialFormer_t2t_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    PartialFormer_t2t_wmt_en_de(args)


@register_model_architecture("PartialFormer", "PartialFormer_t2t_wmt_en_ro_big")
def PartialFormer_t2t_wmt_en_ro_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = 0.3
    PartialFormer_t2t_wmt_en_de(args)
