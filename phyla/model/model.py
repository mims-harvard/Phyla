# from recurrent_memory_transformer_pytorch import RecurrentMemoryTransformer
from typing import List
import torch.nn as nn
import torch
import math
from functools import partial
import json
import os
from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.models.config_mamba import MambaConfig

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import logging
from ..utils.utils import load_config
from skbio import DistanceMatrix
from skbio.tree import nj

#ALL MAMBA CODE FROM: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        #d_model has to be the length of the input sequence which is known

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, pos):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        #x = x + self.pe[:x.size(0)]
        x = x + self.pe[pos].squeeze(2)
        return self.dropout(x)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bidirectional = False,
    bidirectional_strategy = "add",
    bidirectional_weight_tie = True,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }
    if bidirectional:
        mixer_cls = partial(BiMambaWrapper, layer_idx=layer_idx, **ssm_cfg, **bidirectional_kwargs, **factory_kwargs)
    else:
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: str = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None
    
    def forward(self, hidden_states, inference_params=None, cpu = None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")
        return out

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        positional_embeddings: bool = False,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        n_gpus = 1,
        hidden_states = None,
        bidirectional = False,
        bidirectional_strategy = "add",
        bidirectional_weight_tie = True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # if hidden_states is None:
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        if positional_embeddings:
            self.positional_embeddings = True
            self.positional_embeddings = PositionalEncoding(d_model)
        else:
            self.positional_embeddings = False

        self.hidden_states = hidden_states

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bidirectional = bidirectional,
                    bidirectional_strategy = bidirectional_strategy,
                    bidirectional_weight_tie = bidirectional_weight_tie,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    # TODO: Removed StripedHyena code for testing
    # def forward(self, input_ids, inference_params=None, position_ids = None, prev_hiddens = False, seq_reps = None):
    #     # TODO: Take in hidden states if taking in sequence representations (for 2nd or higher [Mamba+attention] block)
    #     if not prev_hiddens:
    #         hidden_states = self.embedding(input_ids)
    #     else: 
    #         hidden_states = seq_reps
    def forward(self, input_ids, inference_params=None, position_ids = None, hidden_states_given = False):
        if hidden_states_given is False:
            hidden_states = self.embedding(input_ids.cuda()) # TODO: Comment out for big model evaluation
            # hidden_states = self.embedding(input_ids)
            if self.positional_embeddings and position_ids is not None:
                hidden_states = self.positional_embeddings(hidden_states, position_ids)
        else:
            hidden_states = input_ids

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states

class Mamba_ModelConfig():
    d_model: int = 256
    n_layer: int = 16
    vocab_size: int = 24
    ssm_cfg: dict = {}
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    num_blocks: int = 40
    model_name: str = 'MAMBA'
    calculation_method: str = "attention"
    positional_embeddings: bool = False
    inject_rotary_attention: bool = False
    bidirectional_strategy: str = "add"
    bidirectional_weight_tie: bool = True
    bidirectional: bool = False
    ranking_loss: bool = False

class Config():
    model: model = Mamba_ModelConfig()

class Phyla(nn.Module):

    amino_acid_encoding = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
                                   "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
                                   "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
                                   "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19}


    def __init__(self, config_path=None, logger = None, name=None, deepspeed = False):
        super().__init__()

        if name is None:
            raise Exception("No name provided must provide a model name, see README for available names")
        elif name == 'phyla-alpha':
            config = Config()
            if config_path: 
                config = load_config(config_path)
        else:
            raise Exception(f"Name {name} not recognized")
        
        modules = []
        modules.append(Mamba_LM_Tree_HeadModel(config.model, hidden_states=True, layer_idx = 0, logger = logger))
        for i in range(config.model.num_blocks-2):
            modules.append(Mamba_LM_Tree_HeadModel(config.model, hidden_states=True, layer_idx= i+1, logger=logger))
        modules.append(Mamba_LM_Tree_HeadModel(config.model, layer_idx= i+1, logger = logger))

        num_blocks = len(modules)
        self.modul = nn.ModuleList(modules)
        self.num_per_block = 5
        
        #Empirically tested and found that 3 blocks fit on a single GPU
        num_gpus = torch.cuda.device_count()
        num_gpus_required = num_blocks // self.num_per_block
        # if num_gpus < num_gpus_required:
        #     raise ValueError(f"Not enough GPUs to run model have {num_gpus} need {num_gpus_required}")
        self.gpu_count = num_gpus-1
        self.num_calls = 0
        self.shard = False # TODO: For 1 GPU evaluation
        self.logger_ = logger
        self.deepspeed = deepspeed
    
    def load(self):
        if 'weights' not in os.listdir():
            os.mkdir('weights')
            os.system("wget https://zenodo.org/records/14657163/files/phyla_alpha_291M_state_dict.pt -P weights")

        path_to_checkpoint = "weights/phyla_alpha_291M_state_dict.pt"
        state_dict = torch.load(path_to_checkpoint)
        #remove _forward_module from each of the keys of state_dict then use that to load the model
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace("_forward_module.model.", "")] = state_dict[key]

        self.load_state_dict(new_state_dict)
        return self
    
    def redistribute_layers(self):
        current_gpu_count = self.gpu_count

        # TODO: Code to avoid the cuda-1 error
        # This should only be ran for evaluation or training with non-deepspeed models
        self.num_per_block = 10
        # print("Num per block: %s" % self.num_per_block)

        num_layer = 0
        for idx, module in enumerate(self.modul):
            self.modul[idx] = module.to(f"cuda:{current_gpu_count}")
            num_layer += 1
            if num_layer%self.num_per_block == 0:
                current_gpu_count -= 1

    def forward(self, x, sequence_mask, cls_token_mask, logits=False):
        final_output_logits = logits
        
        x = self.modul[0](x.to(f'cuda:{self.gpu_count}'), 
                        logits = False, 
                        position_ids = None, 
                        sequence_mask = sequence_mask.to(f'cuda:{self.gpu_count}'),
                        cls_token_mask = cls_token_mask.to(f'cuda:{self.gpu_count}'))  
    
        for module in self.modul[1:-1]:
            correct_device = next(module.parameters()).device
            x = module(x.to(correct_device), 
                    hidden_states_given = True, 
                    logits = False, 
                    position_ids = None, 
                    sequence_mask = sequence_mask.to(correct_device),
                    cls_token_mask = cls_token_mask.to(correct_device))
        
        result = self.modul[-1](x, hidden_states_given = True, 
                                    logits = final_output_logits,
                                    position_ids = None,
                                    sequence_mask = sequence_mask, 
                                    cls_token_mask = cls_token_mask)

        return result

    def reconstruct_tree(self, sequence_embeddings, sequence_names):
        """
        Creates tree from pairwise distance matrix
        Input: (list of [float]) pairwise distance matrix
            (list of str) ids for the matrix
        Output: () reconstructed tree

        From scikit-bio docs: https://scikit.bio/docs/latest/generated/skbio.tree.nj.html#skbio.tree.nj
        """
        distance_matrix = torch.cdist(sequence_embeddings, sequence_embeddings, compute_mode='donot_use_mm_for_euclid_dist').cpu().detach().numpy()[0]
        # Reconstruct tree using scikit bio
        dm = DistanceMatrix(distance_matrix, sequence_names)
        tree = nj(dm)
        return tree

    def encode_fasta(self, fasta_file):
        """
        Encode a fasta file into a tensor of integers that represents the amino acids, positions of cls tokens, and where each sequence begins and ends
        """
        sequences = []
        sequence_names = []
        with open(fasta_file, "r") as file:
            for line in file:
                if line[0] == ">":
                    sequence_names.append(line[1:].strip())
                else:
                    sequences.append(line.strip())

        return self.encode(sequences, sequence_names)
    
    def encode(self, sequences: List[str], sequence_names: List[str]):
        """
        Encode a list of sequences into a tensors of integers that represents the amino acids, positions of cls tokens, and where each sequence begins and ends
        """
        encoded_aa = []
        cls_token_mask = []
        sequence_mask = []
        count = 0

        for seq in sequences:
            encoded_aa.append(22)
            cls_token_mask.append(1)
            sequence_mask.extend([count]*(len(seq)+1))

            for aa in seq:
                if aa not in self.amino_acid_encoding.keys():
                    encoded_aa.append(23)
                    cls_token_mask.append(0)
                else:
                    encoded_aa.append(self.amino_acid_encoding[aa])
                    cls_token_mask.append(0)
            
            count += 1

        return torch.IntTensor(encoded_aa).unsqueeze(0), torch.IntTensor(cls_token_mask).unsqueeze(0).bool(), torch.IntTensor(sequence_mask).unsqueeze(0), sequence_names

class Mamba_LM_Tree_HeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config,
        initializer_cfg=None,
        device=None,
        dtype=None,
        hidden_states = None,
        layer_idx = 0,
        logger = None,
        deepspeed = False

    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            positional_embeddings=config.positional_embeddings,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            device = device,
            dtype = dtype,
            hidden_states = hidden_states,
            bidirectional = config.bidirectional,
            bidirectional_strategy = config.bidirectional_strategy,
            bidirectional_weight_tie = config.bidirectional_weight_tie
        )

        #Hardcoding to 20 amino acids, cause config.model.vocab_size is 24 due to padding, masked AA, and other stuff
        self.lm_head = nn.Linear(d_model, 20, bias=False, **factory_kwargs)
        self.hidden_states = hidden_states
        self.logger_ = logger
        self.deepspeed = deepspeed

        
        if config.calculation_method == 'attention':
            #Took out a .cuda() below here
            self.tree_head = nn.MultiheadAttention(config.d_model, 1, batch_first = True)
            #Adding this new multihead attention here, did not work!
            #self.sequence_head = nn.MultiheadAttention(config.d_model, 1, batch_first = True)
            self.sequence_attention = None
            # self.tree_head_transition = nn.Linear(256, 256, bias = True).cuda()
            # self.tree_head_2 = nn.MultiheadAttention(256, 4, batch_first = True).cuda()

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
        self.tie_weights() 
        self.num_params = sum(p.numel() for p in self.parameters())
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # self.post_norm = RMSNorm(d_model)
        # self.pre_norm = RMSNorm(d_model)
        #self.final_MLP = ParallelGatedMLP(d_model)

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, logits = False, cls_token_mask = False, sequence_mask = False, hidden_states_given = False):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """

        if hidden_states_given:
            hidden_states = input_ids
        else:
            if self.deepspeed:
                failed = False
                try:
                    model_error_tensor = torch.zeros(1).cuda()
                    hidden_states = self.backbone(input_ids, inference_params=inference_params, position_ids = position_ids, hidden_states_given = hidden_states_given)
                except Exception as e:
                    if 'out of memory' in str(e):
                        self.logger_.log(f"ERROR in backbone", level=logging.INFO)
                        model_error_tensor[0] = 1
                    else:
                        self.logger_.log(f"Found a different error and raising it\t{e}", level=logging.INFO)
                        raise e
                finally:
                    torch.distributed.all_reduce(model_error_tensor)
                    if model_error_tensor[0] > 0:
                        self.logger_.log(f"Ooops someone had a OOM we should scuttle", level=logging.INFO)
                        failed = True
                
                if failed:
                    return None
            else:
                hidden_states = self.backbone(input_ids, inference_params=inference_params, position_ids = position_ids, hidden_states_given = hidden_states_given)
        
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        if not logits:
            if self.config.calculation_method == 'attention':
                #hidden_states = self.pre_norm(hidden_states)
                if self.deepspeed:
                    failed = False
                    try:
                        mid_level_error_tensor = torch.zeros(1).cuda()
                        sequence_rep = hidden_states[cls_token_mask].view(cls_token_mask.shape[0], cls_token_mask.sum(dim=1)[0],  -1)
                        memory_rep = hidden_states[~cls_token_mask].view(cls_token_mask.shape[0], (~cls_token_mask).sum(dim=1)[0],  -1)
                        mod_sequence_mask = sequence_mask[~cls_token_mask].view(cls_token_mask.shape[0], (~cls_token_mask).sum(dim=1)[0])
                        #torch.distributed.barrier()
                        #Get rid of -1s
                        unique_values = torch.unique(mod_sequence_mask)
                        unique_values = unique_values[unique_values != -1]
                        x = torch.stack([mod_sequence_mask == value for value in unique_values])
                        x = x.permute(1,0,2)
                        x = ~x
                    except Exception as e:
                        if 'out of memory' in str(e):
                            self.logger_.log(f"ERROR in mid-level here", level=logging.INFO)
                            mid_level_error_tensor[0] = 1
                        else:  
                            self.logger_.log(f"Found a different error and raising it\t{e}", level=logging.INFO)
                            raise e
                    finally:
                        torch.distributed.all_reduce(mid_level_error_tensor)
                        if mid_level_error_tensor[0] > 0:
                            self.logger_.log(f"Ooops someone had a OOM we should scuttle", level=logging.INFO)
                            failed = True
                    
                    if failed:
                        return None
                else:
                    sequence_rep = hidden_states[cls_token_mask].view(cls_token_mask.shape[0], cls_token_mask.sum(dim=1)[0],  -1)
                    memory_rep = hidden_states[~cls_token_mask].view(cls_token_mask.shape[0], (~cls_token_mask).sum(dim=1)[0],  -1)
                    mod_sequence_mask = sequence_mask[~cls_token_mask].view(cls_token_mask.shape[0], (~cls_token_mask).sum(dim=1)[0])
                    #torch.distributed.barrier()
                    #Get rid of -1s
                    unique_values = torch.unique(mod_sequence_mask)
                    unique_values = unique_values[unique_values != -1]
                    x = torch.stack([mod_sequence_mask == value for value in unique_values])
                    x = x.permute(1,0,2)
                    x = ~x

                if self.sequence_attention is not None:
                    memory_rep = self.sequence_attention(memory_rep)[0]

                if self.deepspeed:
                    try:
                        error_tensor = torch.zeros(1).cuda()
                        sequence_rep, weights = self.tree_head(sequence_rep.cuda(), memory_rep.cuda(), memory_rep.cuda(), attn_mask = x.cuda() )
                    except Exception as e:
                        if 'out of memory' in str(e):
                            self.logger_.log(f"ERROR in tree head", level=logging.INFO)
                            error_tensor[0] = 1
                        else:
                            self.logger_.log(f"Found a different error and raising it\t{e}", level=logging.INFO)
                            raise e
                    finally:
                        torch.distributed.all_reduce(error_tensor)
                        if error_tensor[0] > 0:
                            self.logger_.log(f"Ooops someone had a OOM we should scuttle", level=logging.INFO)
                            failed = True
                    
                    if failed:
                        return None
                else:
                    sequence_rep, weights = self.tree_head(sequence_rep.cuda(), memory_rep.cuda(), memory_rep.cuda(), attn_mask = x.cuda() ) # TODO: Comment out for big model evaluation
                    # sequence_rep, weights = self.sequence_head(sequence_rep.cuda(), sequence_rep.cuda(), sequence_rep.cuda())

                    # sequence_rep, weights = self.tree_head(sequence_rep, memory_rep, memory_rep, attn_mask = x)
    
                #print(f"Tree head successful! {torch.distributed.get_rank()}")
                if self.hidden_states:
                    if self.deepspeed:
                        try:
                            error_tensor = torch.zeros(1).cuda()

                            hidden_states = hidden_states.clone()
                            hidden_states[cls_token_mask] = sequence_rep.reshape(-1, sequence_rep.shape[2])
                            hidden_states[~cls_token_mask] = memory_rep.reshape(-1, memory_rep.shape[2])
                            #hidden_states = self.post_norm(hidden_states)
                            #hidden_states = self.final_MLP(hidden_states)
                        except Exception as e:
                            if 'out of memory' in str(e):
                                self.logger_.log(f"ERROR in final layer, it was the cloning", level=logging.INFO)
                                error_tensor[0] = 1
                            else:
                                self.logger_.log(f"Found a different error and raising it\t{e}", level=logging.INFO)
                                raise e
                        finally:
                            torch.distributed.all_reduce(error_tensor)
                            if error_tensor[0] > 0:
                                self.logger_.log(f"Ooops someone had a OOM we should scuttle", level=logging.INFO)
                                failed = True
                        
                        if failed:
                            return None
                    else:
                        hidden_states = hidden_states.clone()
                        hidden_states[cls_token_mask] = sequence_rep.reshape(-1, sequence_rep.shape[2])
                        hidden_states[~cls_token_mask] = memory_rep.reshape(-1, memory_rep.shape[2])

                    return hidden_states

                return sequence_rep # sequence_rep of shape [1, num_seqs, 256]
            return hidden_states
        else:
            return self.lm_head(hidden_states) # hidden_states of shape [1, num_tokens, 256]

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)


