import torch 
import os
import sys
import yaml
import torch
import logging
import inspect 
from itertools import combinations
import torch.nn.functional as F
from skbio import DistanceMatrix
from skbio.tree import nj
from Bio import Phylo
from io import StringIO
from ete3 import Tree

class CustomLogger:
    def __init__(self, log_file, log_to_terminal=False):
        os.makedirs("logs", exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Optionally add a stream handler to log to the terminal
        if log_to_terminal:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(stream_handler)
        
        # Formatter that includes line numbers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the file handler to the logger
        self.logger.addHandler(file_handler)
    
    def log(self, message, level=logging.INFO):

        frame = inspect.currentframe()
        outer_frame = inspect.getouterframes(frame)[1]
        file_name = outer_frame.filename
        line_number = outer_frame.lineno

        try:
            # Modify the message to include the file name and line number
            message = f"[Rank {torch.distributed.get_rank()}]: {message}({file_name.split('/')[-1]}:{line_number})"
        except:
            message = f"{message}({file_name.split('/')[-1]}:{line_number})"

        # Convert the message to a string if it's a dictionary
        if isinstance(message, dict):
            message = json.dumps(message)  # For a JSON-like string
            # message_str = str(message)  # Alternatively, for a simple string conversion
        else:
            message = message

        if level == logging.DEBUG:
            self.logger.debug(message)
        elif level == logging.INFO:
            self.logger.info(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.CRITICAL:
            self.logger.critical(message)

def load_config(Config):
    #Only handles one nested level of config and assumes one nested level
    config = Config()

    try:
        config_filename = sys.argv[1:][0]
    except Exception:
        raise Exception("Must include a config file << python X config_file >>")

    try:
        args = yaml.safe_load(open(config_filename, "r"))
    except Exception:
        raise Exception(f"Config file {config_filename} not found")
    
    for i in args:
        if hasattr(config, i):
            attr = getattr(config, i)
            for j in args[i]:
                if hasattr(attr, j):
                    setattr(attr, j, args[i][j])
                else:
                    raise Exception(f"{j} not in {i} config")
        else:
            raise Exception(f"{i} not in config")
    
    return config

def sample_quartets(n_taxa: int, n_samples: int, *, rng=None):
    """
    Uniformly sample `n_samples` quartets without replacement
    from {0, …, n_taxa-1}.  Falls back to “all quartets” if the
    requested number exceeds  C(n_taxa, 4).
    """
    rng = rng or random
    if n_taxa < 4:
        return torch.empty(0, 4, dtype=torch.long)          # nothing to score
    all_q = list(combinations(range(n_taxa), 4))
    if n_taxa <= 8:
        return torch.tensor(all_q, dtype=torch.long)
    if n_samples >= len(all_q):
        return torch.tensor(all_q, dtype=torch.long)
    return torch.tensor(rng.sample(all_q, n_samples), dtype=torch.long)


def labelled_quartet_loss(D_pred, D_true, quartets, T=0.1):

    def logits_from_sums(s1, s2, s3, T):
        sums = torch.stack([s1, s2, s3], dim=1)          # (Q,3)
        centred = sums - sums.mean(dim=1, keepdim=True)  # zero-mean per quartet
        return -centred / T 
    
    max_p = D_pred.max()
    max_t = D_true.max()
    D_pred = D_pred / (max_p + 1e-8)
    D_true = D_true / (max_t + 1e-8)

    i,j,k,l = quartets.T

    # distance sums
    s_ij_kl = D_pred[i,j] + D_pred[k,l]
    s_ik_jl = D_pred[i,k] + D_pred[j,l]
    s_il_jk = D_pred[i,l] + D_pred[j,k]

    # centred logits
    logits = logits_from_sums(s_ij_kl, s_ik_jl, s_il_jk, T)
                    
    gt_scores = torch.stack([
        D_true[i,j] + D_true[k,l],
        D_true[i,k] + D_true[j,l],
        D_true[i,l] + D_true[j,k],
    ], dim=1)

    target = gt_scores.argmin(dim=1)  

    ce = F.cross_entropy(logits, target, reduction='none')
    ce = ce.clamp(max=5.0)
    return ce.mean()

def soft_quartet_loss(D: torch.Tensor,
                    quartets: torch.Tensor,
                    temperature: float = 0.1) -> torch.Tensor:
    """
    D  : (N, N) pair-wise distance matrix
    quartets : (Q, 4) int64 indices into D
    returns a scalar loss (mean entropy over quartets)
    """
    if quartets.numel() == 0:
        # fewer than 4 taxa ⇒ “no topology signal”; return 0 so the
        # gradient is zero and training can continue
        return D.new_tensor(0.0, requires_grad=True)

    # Ensure indices are on the same device as D
    quartets = quartets.to(D.device)

    i, j, k, l = quartets.T  # shape: (Q,)
    dij_kl = D[i, j] + D[k, l]
    dik_jl = D[i, k] + D[j, l]
    dil_jk = D[i, l] + D[j, k]

    dists = torch.stack((dij_kl, dik_jl, dil_jk), dim=1)  # (Q, 3)
    probs = torch.softmax(-dists / temperature, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

    return entropy  

def batched_quartet_loss(D_pred, D_true,
                        num_quartets: int,
                        temperature: float = 0.1,
                        rng=None) -> torch.Tensor:	

    """
    D_batch : (B, n, n)   – batched distance matrices
    returns a scalar – the mean quartet loss across the batch
    """

    B, n, _ = D_pred.shape
    losses = []
    entropy_losses = []

    for b in range(B):
        D = D_pred[b]                      # (n, n)
        quartets = sample_quartets(n, num_quartets, rng=rng)
        loss_b_entropy = soft_quartet_loss(D, quartets, temperature=temperature)
        loss_b = labelled_quartet_loss(D_pred[b], D_true[b], quartets, T=temperature)
        losses.append(loss_b)
        entropy_losses.append(loss_b_entropy)

    return torch.stack(losses).mean(), torch.stack(entropy_losses).mean()

def reconstruct_tree(matrix, ids, return_str= False):
    """
    Creates tree from pairwise distance matrix
    Input: (list of [float]) pairwise distance matrix
           (list of str) ids for the matrix
    Output: () reconstructed tree

    From scikit-bio docs: https://scikit.bio/docs/latest/generated/skbio.tree.nj.html#skbio.tree.nj
    """

    # Reconstruct tree using scikit bio
    dm = DistanceMatrix(matrix, ids)
    # import pdb; pdb.set_trace()
    tree = nj(dm)
    tree_str = nj(dm, result_constructor=str)
    # print(Tree(str(tree)).get_ascii())
    if return_str:
        return tree_str
    return tree, dm, tree_str

def rf_distance(tree1_str, tree2_str):
    """
    Calculates Robinson-Foulds distance between two trees
    Input: (str) Newick string of tree 1
           (str) Newick string of tree 2
    Output: (int) output Robinson-Foulds distance
    """
    
    # Remove branch distances from the Newick strings of the predicted and reference tree
    def remove_branch_distances(tree_str):

        # Set branch lengths in tree to zero
        phylo_tree = Phylo.read(StringIO(tree_str), "newick")
        for i in phylo_tree.get_nonterminals():
            i.branch_length=None
        for i in phylo_tree.get_terminals():
            i.branch_length=None

        # Convert edited tree to Newick string
        new_str_obj = StringIO()
        Phylo.write(phylo_tree, new_str_obj, "newick")
        new_str_obj.seek(0)
        new_str = new_str_obj.getvalue()

        # Remove distances from edited tree string
        dist_decimals = 8   # To remove the distance value of ":0.00000"
        while True:
            try:
                curr_index = new_str.index(":")
                new_str = new_str[:curr_index] + new_str[curr_index+dist_decimals:]
            except:
                return new_str

    tree1_str_nodist = remove_branch_distances(tree1_str)
    tree2_str_nodist = remove_branch_distances(tree2_str)

    # Calculate tree comparison metrics
    t1 = Tree(tree1_str_nodist)
    t2 = Tree(tree2_str_nodist)
    result = t1.compare(t2, unrooted=True)
    rf = int(result["rf"])
    max_rf = int(result["max_rf"])
    norm_rf = result["norm_rf"]
    
    return {"rf": rf,
            "max_rf": max_rf,
            "norm_rf": norm_rf}
