from dataset.data import OpenFold_Dataset
from model.model import  Phyla
import torch
from utils.utils import load_config, CustomLogger
from utils.eval_configs import Mamba_ModelConfig, DatasetConfig, TrainerConfig
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from run.TrainingModule import TrainingModule
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb


pl.seed_everything(42) 

class Config():
    model: model = Mamba_ModelConfig()
    dataset: dataset = DatasetConfig()
    trainer: trainer = TrainerConfig()

"""
Run this script using "python3 -m run.run configs/config.yaml" while in the home directory

"""
def train_phyloLLM(config):
    logger = CustomLogger(log_file=f"logs/phyla_training_{config.trainer.run_name}.log", log_to_terminal=True)

    dparam = {
        "dataset_directories": config.dataset.dataset_directories,
        "logger": logger,
        "dataset_size": config.dataset.dataset_size,
    }

    dataset = OpenFold_Dataset(**dparam)

    model = Phyla(config)

    hparams = {'lr': config.trainer.lr,
               'record': config.trainer.record,
               'lr_scheduler': config.trainer.scheduler,
               'num_annealing_steps': config.trainer.num_annealing_steps,
               'num_warmup_steps': config.trainer.num_warmup_steps,
               'dataset': dataset,
               'logger': logger,
               'use_mlm_loss': config.trainer.use_mlm_loss,
               'use_quartet_loss': config.trainer.use_quartet_loss}

    model = TrainingModule(model, **hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    
    trainer_args = {}
    if config.trainer.record:
        wandb.init(project = 'genome_llms')
        wandb.watch(model, log_freq=100)
    
    trainer_args['max_epochs'] = config.trainer.epochs

    save_callback = ModelCheckpoint(
        dirpath=config.trainer.save_path,
        save_top_k=1,  # Save only the best checkpoint
        monitor="treefam_avg_normrf",  # Metric to monitor
        mode="min",  # Save the checkpoint with the minimum value of the metric
        filename="{epoch:02d}-{step:06d}-{treefam_avg_normrf:.4f}",  # Include metric value in the filename
        save_last=True  # Optionally save the last checkpoint as well
    )

    # Also save model every config.trainer.steps_callback steps
    save_callback2 = ModelCheckpoint(
        dirpath=config.trainer.save_path,
        filename="{epoch:02d}-{step:06d}-{treefam_avg_normrf:.4f}",  # Include metric value in the filename
        every_n_train_steps=config.trainer.steps_callback,  # Save every N steps
        save_top_k=-1  # Save all checkpoints
    )

    trainer_args['callbacks'] = [save_callback, save_callback2] # For validation callback runs

    if config.trainer.val_callback_freq != 0:
        trainer_args['val_check_interval'] = config.trainer.val_callback_freq

    trainer_args['accelerator'] = "gpu"
    
    trainer = Trainer(**trainer_args)

    prev_checkpoint = None

    if not prev_checkpoint:
        trainer.fit(model, train_dataloaders = dataset.train_dataloader(), val_dataloaders = dataset.val_dataloader())
    elif prev_checkpoint:
        trainer.fit(model, train_dataloaders = dataset.train_dataloader(), val_dataloaders = dataset.val_dataloader(), ckpt_path=prev_checkpoint)

if __name__ == "__main__":
    config = load_config(Config) 
    train_phyloLLM(config)