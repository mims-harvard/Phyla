from dataset.data import Protein_Dataset, OpenFold_Dataset
from model.model import  Phyla
import torch
from utils.utils import load_config
from pytorch_lightning import Trainer
import pytorch_lightning as pl

pl.seed_everything(42) 

class DatasetConfig():
    dataset: str = '40'
    #Number of sub trees per batch
    batch_size: int = 2
    #Number of sequences per sub tree
    sub_tree_size: int = 10
    #Detect what gpu you are on, make tree/trees in range of 0 to largest sub-tree size for that gpu
    adaptive_batch_size: bool = False
    max_subtree_size_scaler: int = 1
    new_construction: bool = False

class Mamba_ModelConfig():
    d_model: int = 256
    n_layer: int = 8
    vocab_size: int = 24
    ssm_cfg: dict = {}
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    num_blocks: int = 1
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
    dataset: dataset = DatasetConfig()
    trainer: trainer = TrainerConfig()

"""
Run this script using "python3 -m run.run configs/config.yaml" while in the home directory

"""
def train_phyloLLM(config):

    # data = #Some data converter for the model

    model = Phyla(config)

    hparams = {'lr': config.trainer.lr, 
    'record': config.trainer.record,
     'mode':config.model.model_name, 
     'calculation_method': config.model.calculation_method,
     'lr_scheduler': config.trainer.scheduler,
     'num_annealing_steps': config.trainer.num_annealing_steps,
     'num_warmup_steps': config.trainer.num_warmup_steps,
     'dataset': dataset,
     'deepspeed': config.trainer.deepspeed,
     'logger': logger,
     'use_tree_loss': config.trainer.use_tree_loss,
     'symmetry_loss': config.trainer.symmetry_loss,
     'ranking_head': ranking_head,}

    model = TrainingModule(model, **hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # import pdb; pdb.set_trace()
    trainer = Trainer(**trainer_args)

    prev_checkpoint = "Path"

    if not prev_checkpoint:
        trainer.fit(model, dataset.train_dataloader())
    elif prev_checkpoint:
        trainer.fit(model, dataset.train_dataloader(), ckpt_path=prev_checkpoint)

if __name__ == "__main__":
    config = load_config(Config) 
    train_phyloLLM(config)
