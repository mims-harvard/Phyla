import torch, torch.nn as nn, torch.optim as optim, numpy as np
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import wandb
import signal
import logging
import gc
import torch.distributed
from skbio import DistanceMatrix
from skbio.tree import nj
from utils.utils import  batched_quartet_loss, reconstruct_tree, rf_distance
import random
import gc
from scipy.stats import rankdata
import torchsort
import torch
import random
import math
from itertools import combinations
from eval.evo_reasoning_eval import tree_reconstruction_benchmark

class TrainingModule(LightningModule):
	def __init__(
		self,
		model =  None,
		lr = 1e-4,
		record = False,
		epochs = 5000,
		lr_scheduler = 'default',
		num_annealing_steps = 10000,
		num_warmup_steps = 1000,
		dataset = None,
		logger = None,
		use_mlm_loss = True,
		use_quartet_loss = True,
	):
		super().__init__()
		self.model = model
		self.lr = lr
		self.record = record
		self.model_name = model
		self.epochs = epochs
		self.warmup_steps = 400
		self.current_step_value = 0
		self.lr_scheduler = lr_scheduler
		self.num_annealing_steps = num_annealing_steps
		self.num_warmup_steps = num_warmup_steps
		self.dataset = dataset
		self.use_mlm_loss = use_mlm_loss
		self.use_quartet_loss = use_quartet_loss

		# Important: This property activates manual optimization.
		# Turning off automatic optimization so I can catch out of memory errors!
		self.automatic_optimization = False
		self.logger_ = logger

	def forward(
		self,
		batch,
		logits = False,
		cls_token_mask = False,
		sequence_mask = None,
		cache = None
	):

		return self.model(batch, logits = logits,
					 cls_token_mask = cls_token_mask,
					 sequence_mask = sequence_mask)

	def mlm_accuracy(self, predicted_AA, real_AA, mask=None):
		res = (predicted_AA.detach().cpu()==real_AA.cpu())
		return res.sum()/res.shape[0]

	def get_sequence_embeddings(self, encoded_sequences, cls_positions, sequence_mask):
		return self.forward(encoded_sequences, None, logits = False, 
											cls_token_mask = cls_positions.bool(), 
											sequence_mask  = sequence_mask)

	def compute_rf_distance(self, pairwise_distances, tree_matrix, tree_labels):
		rf_raw_res = []
		rf_max_res = []
		rf_norm_res = []

		for pred_matrix, real_matrix, tree_labels in zip(pairwise_distances, tree_matrix, tree_labels):
			pred_tree = reconstruct_tree(pred_matrix, tree_labels, return_str = True)
			real_tree = reconstruct_tree(real_matrix, tree_labels, return_str = True)
			rf = rf_distance(pred_tree, real_tree)
			rf_raw_res.append(rf['rf'])
			rf_max_res.append(rf['max_rf'])
			rf_norm_res.append(rf['norm_rf'])

		return rf_raw_res, rf_norm_res, rf_max_res

	def step(self, batch, eval = False):
		"""
		There is a bug here where we reconstruct tree from masked sequences but we fix once it is a problem
		"""

		# Calculate MLM loss

		logs = {}
		if self.use_mlm_loss:
			logits = self.forward(batch['masked_sequences'].cuda(), None, logits = True, 
						sequence_mask  = batch['sequence_mask'].cuda(), 
						cls_token_mask = batch['cls_positions'].bool().cuda())
			if logits is None:
				return None
			logits = logits[batch['masked_positions'].bool()]
			real_AA = batch['masked_identities'][batch['masked_positions'].bool()]
			AA_probs = torch.softmax(logits, 1)
			real_probs = torch.nn.functional.one_hot(real_AA.long(), num_classes=24)
			if torch.cuda.is_available():
				real_probs = real_probs.cuda()
			mlm_loss = nn.CrossEntropyLoss()(AA_probs, real_probs.float())
			mlm_accuracy = self.mlm_accuracy(AA_probs.argmax(axis=-1), real_AA)
			
			self.logger_.log(f"{mlm_loss}\t{mlm_accuracy}", level=logging.INFO)

			if eval:
				logs['mlm_loss'] = mlm_loss.detach().cpu()
			else:
				logs['mlm_loss'] = mlm_loss

			logs['mlm_accuracy'] = mlm_accuracy
		sequence_representations = self.forward(batch['true_sequences'], logits = False, 
									cls_token_mask = batch['cls_positions'].bool(), 
									sequence_mask  = batch['sequence_mask'])
		
		if sequence_representations is None:
			return None

		sequence_representations = sequence_representations.to(dtype=torch.float32)  # Convert to float32

		pairwise_distances = torch.cdist(sequence_representations, sequence_representations)
		if torch.cuda.is_available():
				batch['tree_matrix'] = batch['tree_matrix'].cuda()
				pairwise_distances = pairwise_distances.cuda()

		self.logger_.log(f"Calculating quartet loss", level=logging.INFO)
		n_taxa = pairwise_distances.size(1)               # second dim!
		batch_rng  = random.Random(42)
		num_q  = min(100, int(n_taxa * math.log2(max(n_taxa, 4))))
		quartet_loss_value, entropy_loss_value = batched_quartet_loss(pairwise_distances, batch['tree_matrix'],
												num_q,
												temperature=0.1,
												rng=batch_rng)
		
		logs['quartet_loss'] = quartet_loss_value
		logs['entropy_quartet_loss'] = entropy_loss_value
		
		# Initialize the loss
		logs['loss'] = 0

		if self.use_mlm_loss:
			logs['loss'] += logs['mlm_loss']
		else:
			logs['mlm_loss'] = torch.tensor(0)

		if self.use_quartet_loss:
			logs['quartet_loss'] = quartet_loss_value
			logs['loss'] += logs['quartet_loss']
		else:
			logs['quartet_loss'] = torch.tensor(0)


		self.logger_.log(f"Starting RF Metric Calculating", level=logging.INFO)
		rf_raw, rf_norm, rf_max = self.compute_rf_distance(torch.cdist(sequence_representations.detach().cpu(), sequence_representations.detach().cpu(), compute_mode='donot_use_mm_for_euclid_dist').numpy(), 
								batch['tree_matrix'].cpu().numpy(), 
								batch['tree_labels'])

		print(f"RF NORM {rf_norm}\t{np.mean(rf_norm)}")

		self.logger_.log(f"RF Metric Calculated", level=logging.INFO)
		logs['norm_rf_distance'] = torch.tensor(sum(rf_norm)/len(rf_norm))

		return logs
			
		
	def training_step(self, batch, _):
		opt = self.optimizers()
		opt.zero_grad()


		success = False
		num = 0

		#Logic, if we have an out of memmory error we just resample with a smaller subtree and rerun
		while not success:

			opt.zero_grad()
			if num > 1:
				print("Batch is too large decreasing max tree and number of subtrees by a factor of 1.2")
				index, sub_tree_size, num_subtrees = self.dataset.chosen_tree
				new_sub_tree_size = sub_tree_size
				new_num_subtrees = int(num_subtrees//1.2)

				if new_num_subtrees == 0:
					new_num_subtrees = 1
					new_sub_tree_size = int(sub_tree_size//1.2)
				
				if new_sub_tree_size < 5:
					new_sub_tree_size = 5
					new_num_subtrees = 1
				
				if num <= 2:
					new_max_aa = new_num_subtrees*new_sub_tree_size * self.dataset.return_max_length(self.dataset.name_to_seq)
					print(f"Updating the adaptive batch size sampler with this new information of the max aa of {new_max_aa}")
					self.dataset.size_detector.update_max_aa(new_max_aa)
				
				if num > 10:
					print("We are spiraling, moving on")
					return torch.tensor(0)
				
				sub_batch = self.dataset.__getitem__(index, preset_subtree_size = new_sub_tree_size)
				batch  = self.dataset.collate_fn([sub_batch], preset_subtree_num = new_num_subtrees)
				print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
				print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

				gc.collect()

			try:
				print(f"Memory allocated before step: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
				print(f"Memory reserved before step: {torch.cuda.memory_reserved() / 1024 ** 2} MB")


				logs = self.step(batch)
				loss = logs['loss']

				print(f"Memory allocated before backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
				print(f"Memory reserved before backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

				self.manual_backward(loss)
				success = True
				failed  = False

				print(f"Memory allocated after backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
				print(f"Memory reserved after backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

			except RuntimeError as e:
				if 'out of memory' in str(e):
					print('WARNING: out of memory')
					if hasattr(torch.cuda, 'empty_cache'):
						torch.cuda.empty_cache()

					print(f"Memory allocated after OOM: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
					print(f"Memory reserved after OOM: {torch.cuda.memory_reserved() / 1024 ** 2} MB")	

					num += 1
				else:
					raise e

		if not failed and logs is not None:
			for k, v in logs.items():        
				self.log(
							k, v.to("cuda"), on_step=True, on_epoch=False, prog_bar=True, logger=True,
							sync_dist=True
							)

			index, sub_tree_size, num_subtrees = self.dataset.chosen_tree
			lr = opt.optimizer.param_groups[0]["lr"]
			self.log('num_seq_per_subtree', sub_tree_size)
			logs['num_seq_per_subtree'] = sub_tree_size
			self.log('num_subtrees', num_subtrees)
			logs['num_subtrees'] = num_subtrees
			self.log('lr', lr)
			logs['lr'] = lr
			self.logger_.log(logs, level=logging.INFO)

		if logs is not None:
			if self.record:
				wandb.log(logs, step=self.global_step)

			self.clip_gradients(
				opt,
				gradient_clip_val=1.0,             # tighten / loosen here
				gradient_clip_algorithm="norm"
			)

			self.current_step_value += 1
			opt.step()

			# Perform learning rate schedling
			if self.lr_scheduler == "cosine":
				sch1 = self.lr_schedulers()
				sch1.step()
			elif self.lr_scheduler == "cosine_warmup":
				sch1, sch2 = self.lr_schedulers()
				# Perform warmup
				if self.num_warmup_steps > 0:
					sch1.step()
					self.num_warmup_steps -= 1
				# Perform cosine annealing
				else:
					sch2.step()
			elif self.lr_scheduler == "warmup":
				sch1 = self.lr_schedulers()
				# Perform warmup
				if self.num_warmup_steps > 0:
					sch1.step()
					self.num_warmup_steps -= 1

			return logs['loss']
		else:
			return torch.tensor(0)

	def testing_step(self, batch, _):
		pass

	def epoch_end(self, outputs, stage_str):
		pass

	def on_training_epoch_end(self, outputs):
		return self.epoch_end(outputs, 'Train')

	def validation_step(self, batch, batch_idx):
		self.eval()
		models = {}
		models["PHYLA"] = {"model": self, "alphabet_tokenizer": None}
		last_dataset_id = 1533
		num_datasets = [0, last_dataset_id]
		dataset = "treebase"
		output_file = None

		for dataset in ["treefam", "treebase"]:
			normrfs, tree_sizes = tree_reconstruction_benchmark(models, num_datasets, output_file, dataset)
			normrfs = [i for i in normrfs if i is not None]
			avg_normrf = sum(normrfs)/len(normrfs)
			if dataset == "treefam":
				self.log("treefam_avg_normrf", avg_normrf)
				self.log("num_fam_successes", len(normrfs))
			print(f"AVG {dataset} NORMRF: {avg_normrf}")

			if self.record:
				wandb.log({f"{dataset}_avg_normrf": avg_normrf}, step=self.global_step)

		self.train()


	def on_before_optimizer_step(self, optimizer):
		# Compute the 2-norm for each layer
		# If using mixed precision, the gradients are already unscaled here
		norms = grad_norm(self, norm_type=2)
		total = norms['grad_2.0_norm_total']

		layer_norms = {k: v for k, v in norms.items() if "total" not in k}
		max_grad = max(layer_norms.values())
		mean_grad = torch.mean(torch.stack(list(layer_norms.values())))

		self.log("grad_norm_max", max_grad, prog_bar=True, on_step=True)
		self.log("grad_norm_mean", mean_grad, prog_bar=False, on_step=True)

		# Optional: Print a warning if exploding
		if max_grad > 1:
			print(f"[Warning] Gradient norm unusually high: max={max_grad:.2e}, mean={mean_grad:.2e}")

		self.log("grad_norm_total",total)
		print(f"step {self.global_step:4d}  total_grad_norm = {total:.2f} mean is {mean_grad:.2f} max is {max_grad:.2f}")
		if self.record:
			wandb.log({"grad_norm_total": total}, step=self.global_step)
			wandb.log({"grad_norm_max": max_grad}, step=self.global_step)
			wandb.log({"grad_norm_mean": mean_grad}, step=self.global_step)


	def on_before_optimizer_step(self, optimizer):

		norms = grad_norm(self, norm_type=2)
		total = norms['grad_2.0_norm_total']

		layer_norms = {k: v for k, v in norms.items() if "total" not in k}
		max_grad = max(layer_norms.values())
		mean_grad = torch.mean(torch.stack(list(layer_norms.values())))

		self.log("grad_norm_max", max_grad, prog_bar=True, on_step=True)
		self.log("grad_norm_mean", mean_grad, prog_bar=False, on_step=True)

		if max_grad > 1:
			print(f"[Warning] Gradient norm unusually high: max={max_grad:.2e}, mean={mean_grad:.2e}")

		self.log("grad_norm_total",total)
		print(f"step {self.global_step:4d}  total_grad_norm = {total:.2f} mean is {mean_grad:.2f} max is {max_grad:.2f}")
		if self.record:
			wandb.log({"grad_norm_total": total}, step=self.global_step)
			wandb.log({"grad_norm_max": max_grad}, step=self.global_step)
			wandb.log({"grad_norm_mean": mean_grad}, step=self.global_step)

	def test_step(self, batch, batch_idx):
		pass

	def on_test_epoch_end(self, outputs):
		pass

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=self.lr)

		if self.lr_scheduler == 'cosine':
			sch1 = CosineAnnealingLR(optimizer, T_max=self.num_annealing_steps) # Set to current number of steps for training 7 days
			return [optimizer], [sch1]
		elif self.lr_scheduler == "cosine_warmup":
			sch1 = LinearLR(optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps)
			sch2 = CosineAnnealingLR(optimizer, T_max=self.num_annealing_steps)
			return [optimizer], [sch1, sch2]
		elif self.lr_scheduler == "warmup":
			sch1 = LinearLR(optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps)
			return [optimizer], [sch1]
		else:
			scheduler = []
			return optimizer
		