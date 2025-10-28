from phyla import phyla
from configs.eval_configs import *
from configs.config_util import load_config
from pytorch_lightning import LightningModule
from collections import OrderedDict
from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree
import os
import torch
from Bio import Phylo
from Bio.Seq import Seq
from Bio.SeqUtils import IUPACData
from tqdm import tqdm
import pickle
from io import StringIO
import pandas as pd

# TODO: Imports commented out below unnecessary for cleaned Phyla
import esm
# from evo import Evo
# from eval.TreeCluster import TreeCluster
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from statistics import mean
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics.cluster import normalized_mutual_info_score
from eval.TreeCluster import run_treecluster

import time
# import umap
# import matplotlib.pyplot as plt
import numpy as np
from re import sub
from phyla.dataset.data import Arbitrary_Sequence_Dataset
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

"""
Run script with command: python -m eval.evo_reasoning_eval configs/sample_eval_config.yaml 
"""

import pytorch_lightning as pl
np.random.seed(0)
torch.manual_seed(0)
pl.seed_everything(42) 

def load_model(checkpoint_file = None, config = None, random_model = False, device = 'cuda:0'):
    
    if 'Phyla' in config.model_name: 
        model = phyla(name=config.model_name, device = device).load()
        alphabet = None

    elif config.model_name == "ESM2":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    
    elif config.model_name == "EVO":
        evo_model = Evo('evo-1-131k-base')
        model, tokenizer = evo_model.model, evo_model.tokenizer

    elif config.model_name == "ESM3":
        model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
        alphabet = None
    
    elif config.model_name == "ESM2_3B":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    
    else:
        raise Exception("Could not find model!")

    model.eval()
    # if torch.cuda.is_available():
    #     model = model.cuda()

    if "Phyla" in config.model_name:
        return {"model": model, "alphabet_tokenizer": None}
    elif "ESM" in config.model_name:
        return {"model": model.to(device), "alphabet_tokenizer": alphabet}
    elif config.model_name == "EVO":
        return {"model": model, "tokenizer": tokenizer}
    
def generate_tree(seq_file, 
        tree_file, 
        model, 
        alphabet_tokenizer, 
        model_name, 
        dataset_type, 
        eval_mode, 
        save = False, 
        name = None, 
        convert_to_aa = False, 
        dictionary_data = None, 
        random = False, 
        device = 'cuda:0'):

    sequences = []
    names = []
    labels_binary = []
    labels_cont = []

    if dataset_type in ["protein_gym"]:
        seq_id_counter = 0
        for i in open(seq_file).readlines():
            # Skip header
            if seq_id_counter == 0:
                seq_id_counter += 1
                continue
            data = i.split(',')
            sequences.append(data[1].rstrip())
            padding_zeros = (4 - len(str(seq_id_counter)))*"0"
            names.append("seq%s%s" % (padding_zeros, seq_id_counter)) # Generate unique id for each sequence
            labels_cont.append(float(data[2]))
            labels_binary.append(int(data[3]))
            seq_id_counter += 1
        # Format for later RF calculation
        # f = open(tree_file, "r").read()
        # ref_tree_str = "".join([val.split("_")[1] if len(val.split("_"))==2 else val for val in f.split("\n")])
        ref_tree_str = None
    
    elif dataset_type in ["treebase"]:
        nucleotide_letters = {"A", "C", "G", "T", "N"}
        
        for i in open(seq_file).readlines():
            if i[0] == ">":
                names.append(i.strip()[1:])
            else:
                # Convert all lower case values to null values of "."
                curr_seq = sub("[a-z]", '.', i.strip())
                # Check if the current sequence is a protein
                non_nucleotide_count = sum(1 for char in curr_seq if char not in nucleotide_letters)
                non_nucleotide_proportion = non_nucleotide_count / len(curr_seq)
                # If current sequence not a protein, convert to amino acids
                if non_nucleotide_proportion < 0.5:
                    # Add trailing "N" characters to make the sequence length a multiple of 3
                    remainder = len(curr_seq) % 3
                    if remainder != 0:
                        curr_seq += "N" * (3 - remainder)
                    # Convert periods to amino acids
                    curr_seq = curr_seq.replace(".", "N")
                    curr_seq = curr_seq.replace("X", "N")
                    # Convert to amino acids
                    curr_seq = str(Seq(curr_seq).translate())
                    # Manually remove all stop codons from the sequence, as this is outside the vocabulary of PLMs
                    curr_seq = curr_seq.replace("*", "")

                sequences.append(curr_seq)
                
        f = open(tree_file, "r").read()
        # Format for later RF calculation
        ref_tree_str = "".join([val.split("_")[1] if len(val.split("_"))==2 else val for val in f.split("\n")])
        labels_cont = torch.Tensor([])
        labels_binary = torch.Tensor([])

    elif dictionary_data is not None and dataset_type == "treefam":

        for i in dictionary_data["sequences"]:
            names.append(i)
            sequences.append(dictionary_data["sequences"][i].replace('-', ''))
        
        ref_tree_str = dictionary_data["tree_newick"]

        tree_data = ref_tree_str.split('\n')
        name_mapping = {}
        to_replace = []
        for item in tree_data:
            collected = item.split(' ')
            if collected[0] == 'SEQ':
                if ':' in collected[2]:
                    replacemante = [collected[2]]
                    collected[2] = collected[2].replace(':', '-')
                    replacemante.append(collected[2])
                    to_replace.append(replacemante)

                #name_mapping[collected[2]] = f'{collected[1]}|{collected[2]}'
                name_mapping[collected[2]] = f'{collected[2]}'
            elif item[0] == '(':
                ref_tree_str = item

        for i, z in to_replace:
            ref_tree_str = ref_tree_str.replace(i, z)
        
        revised_names = []
        for i in names:
            revised_names.append(i.replace(':', '-'))
        names = revised_names

        tree = Phylo.read(StringIO(ref_tree_str), "newick")

        # Relabel tips
        for leaf in tree.get_terminals():
            if leaf.name in name_mapping:
                leaf.name = name_mapping[leaf.name]
            else:
                raise Exception("NAME MAPPING FAILED IN TREEFAM ")
        
        ref_tree_str = tree.format("newick")

    dataset = Arbitrary_Sequence_Dataset()
    batch, names = dataset.encode_sequences(sequences, names)

    # For finding max batch size for ESM2 and EVO

    max_aa_dict = {"ESM2": 19532, "EVO": 62500} # Derived from papers


    if "Phyla" in model_name:
        with torch.no_grad():
            sequence_embeddings = model(batch['encoded_sequences'].to(device), 
                                        batch['sequence_mask'].to(device),
                                        batch['cls_positions'].bool().to(device))

    elif model_name == "ESM2" or model_name == "ESM2_3B":

        # Tokenize sequences for input to ESM2
        batch_converter = alphabet.get_batch_converter()
        data = [(names[i], sequences[i]) for i in range(len(names))]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Generate token representations in batched sequence format
        token_representations = torch.Tensor([])
        num_seqs = len(batch_lens)
        seq_length = batch_tokens.shape[1]
        seqs_per_chunk = (max_aa_dict["ESM2"] // seq_length)
        token_representations = torch.Tensor([])
        for i in range(0, len(batch_tokens), seqs_per_chunk):
            with torch.no_grad():
                results = model(batch_tokens[i:i+seqs_per_chunk].to(device), repr_layers=[33], return_contacts=False)
            token_representations = torch.cat((token_representations, results["representations"][33].detach().cpu()))

        # Generate sequence embeddings
        sequence_embeddings = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_embeddings.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        sequence_embeddings = torch.stack(sequence_embeddings).unsqueeze(0)

    elif model_name == "EVO":
        sequences = [seq.replace(".", "L") for seq in sequences]
        sequences = [reverse_translate(seq) for seq in sequences]
        # Generate sequence embeddings in batches
        logits = torch.Tensor([])
        num_seqs = len(sequences)

        # For manual padding to deal with different max sizes in sub-batches
        max_seq_length = max([len(seq) for seq in sequences])
        seqs_per_chunk = (max_aa_dict["EVO"] // max_seq_length)
        device = 'cuda:0'
        for i in range(0, num_seqs, seqs_per_chunk):
            with torch.no_grad():
                input_ids, seq_lengths = prepare_batch(sequences[i:i+seqs_per_chunk], alphabet_tokenizer, prepend_bos=False, device=device)
                # Manually pad
                curr_num_to_pad = max_seq_length - input_ids.shape[1]
                input_ids = torch.cat((input_ids, torch.ones(input_ids.shape[0], curr_num_to_pad, dtype=torch.int64).cuda()),dim=1)
                # input_ids = torch.cat((input_ids, torch.ones(input_ids.shape[0], curr_num_to_pad, dtype=torch.int64, device=device)),dim=1)
                results, _ = model(input_ids)
            # TODO: Logits might not be correct, may need embeddings
            logits = torch.cat((logits, results.detach().cpu()))
        sequence_embeddings = logits.mean(1).unsqueeze(0)
    
    elif model_name == "ESM3":
        
        # Calculate sequence embeddings 
        sequence_embeddings = []
        for sequence in sequences:
            protein = ESMProtein(sequence=sequence)
            protein_tensor = model.encode(protein) 
            output = model.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
            token_embeddings = output.per_residue_embedding
            sequence_embeddings.append(token_embeddings.mean(0).unsqueeze(0))
        sequence_embeddings = torch.cat(sequence_embeddings).unsqueeze(0)
    
    elif model_name in ["ESMC_300M", "ESMC_600M"]:

        # Calculate sequence embeddings 
        max_seq_length = max([len(seq) for seq in sequences]) + 2 # Adding +2 because of the built-in ESMC tokenization strategy
        sequence_embeddings = []
        for sequence in sequences:
            protein = ESMProtein(sequence=sequence)
            protein_tensor = model.encode(protein)

            # Manually pad
            curr_num_to_pad = max_seq_length - len(protein_tensor)
            protein_tensor.sequence = torch.cat((protein_tensor.sequence, torch.ones(curr_num_to_pad, dtype=torch.int64).cuda()))

            logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            sequence_embeddings.append((logits_output.embeddings).squeeze().mean(0).unsqueeze(0))
        sequence_embeddings = torch.cat(sequence_embeddings).unsqueeze(0)
    
    elif model_name == "EVO2":
        
        # sequence_embeddings = []
        # for sequence in sequences:
        #     input_ids = torch.tensor(alphabet_tokenizer.tokenize(sequence), dtype=torch.int).unsqueeze(0).to('cuda:0')
        #     layer_name = 'blocks.28.mlp.l3'
        #     outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
        #     sequence_embeddings.append(embeddings[layer_name].squeeze().mean(0).unsqueeze(0))
        # sequence_embeddings = torch.cat(sequence_embeddings).unsqueeze(0)
        # sequence_embeddings = sequence_embeddings.to(torch.float64)

        dataset_batch_size = 10

        from torch.cuda import OutOfMemoryError
        import gc

        def get_embeddings(sequences, batch_size):
            sequence_embeddings = []
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                input_ids = [
                    torch.tensor(alphabet_tokenizer.tokenize(seq), dtype=torch.int)
                    for seq in batch
                ]
                input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
                input_ids = input_ids.to('cuda:0')

                layer_name = 'blocks.28.mlp.l3'
                outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
                emb = embeddings[layer_name].mean(dim=1)  # mean over sequence length
                sequence_embeddings.append(emb)
            return torch.cat(sequence_embeddings, dim=0).unsqueeze(0).to(torch.float64)

        try:
            sequence_embeddings = get_embeddings(sequences, batch_size=dataset_batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM at batch_size=2, retrying with batch_size=1")
                torch.cuda.empty_cache()
                gc.collect()
                dataset_batch_size = 1
                sequence_embeddings = get_embeddings(sequences, batch_size=dataset_batch_size)
            else:
                raise e
    elif model_name == "PROGEN2_XLARGE" or model_name == "PROGEN2_LARGE":
        sequence_embeddings = []
        for seq in sequences:
            target = torch.tensor(alphabet_tokenizer.encode(seq).ids).cuda()
            hidden_states =  model(target, labels=target, hidden_state_override=True).detach().cpu()
            sequence_embeddings.append(hidden_states[1:-1].mean(dim=0))
        sequence_embeddings = torch.stack(sequence_embeddings).float().unsqueeze(0).detach().cpu()

    # Calculate linear probe value if evaluting Tier 2
    if dataset_type in ["protein_gym1", "protein_gym2", "protein_gym"]:
        # Train linear probe on output embeddings
        # TODO: Temp comment out MLPRegressor to calculate just Tier 2
        just_tier2 = True
        if just_tier2:
            corr_coeff_dict = {"spearman": 0,
                                "pearson": 0,
                                "kendall": 0}
        else:
            linear_probe = MLPRegressor(random_state=1, max_iter=500).fit(sequence_embeddings.detach().cpu().squeeze(), torch.Tensor(labels_cont))

            preds = linear_probe.predict(sequence_embeddings.detach().cpu().squeeze())

            # Calculate correlation coefficients
            spearman_res = spearmanr(torch.Tensor(labels_cont), torch.Tensor(preds))
            pearson_res = pearsonr(torch.Tensor(labels_cont), torch.Tensor(preds))
            kendall_res = kendalltau(torch.Tensor(labels_cont), torch.Tensor(preds))
            corr_coeff_dict = {"spearman": spearman_res.statistic,
                            "pearson": pearson_res.statistic,
                            "kendall": kendall_res.statistic}
        
                                                                                                
    elif dataset_type in ["openfold", "openfold_small", "openfold_3tree", "swisstree", "treebase", "treefam"]:
        corr_coeff_dict = {}

    # PLot and save UMAP of embeddings
    if eval_mode:
        proteingym_dataset_name = seq_file.split("/")[-1]
        reducer = umap.UMAP(n_jobs=1, random_state=42)
        umap_out = reducer.fit_transform(sequence_embeddings.squeeze().detach().cpu())
        plt.scatter(umap_out[:,0], umap_out[:,1])
        plt.title("%s %s UMAP" % (proteingym_dataset_name, model_name))
        plt.savefig("eval/figures/%s_%s_umap.png" % (proteingym_dataset_name, model_name))
        plt.clf()
        return

    pred_matrix = torch.cdist(sequence_embeddings, sequence_embeddings, compute_mode='donot_use_mm_for_euclid_dist')
    pred_tree, dm, pred_tree_str = reconstruct_tree(pred_matrix.cpu().detach().numpy()[0], names)
    if save:
        open(f'{name}.nwk', 'w').write(tree_str)
        x = open(f'{name}_labels', 'w')
        for i in open(seq_file).readlines():
            data = i.split(',')
            x.write(f'{data[1]}\t{data[2].rstrip()}\n')
            # names.append(data[1].rstrip())
            # sequences.append(data[0])

    # Calculate pairwise distance matrix (for Tier 2 task) 
    pairwise_distances = {}
    max_dist = 0 # Used for threshold value in clustering
    for i in range(len(dm.ids)):
        if dm.ids[i] not in pairwise_distances:
            pairwise_distances[dm.ids[i]] = {}
        for j in range(len(dm.ids)):
            # Update maximum distance
            if dm[i,j] > max_dist:
                    max_dist = dm[i,j]
            if dm.ids[j] not in pairwise_distances[dm.ids[i]]:
                pairwise_distances[dm.ids[i]][dm.ids[j]] = [dm[i,j]]
            else:
                pairwise_distances[dm.ids[i]][dm.ids[j]].append(dm[i,j])

    return {"pred_tree": pred_tree,
            "pred_tree_str": pred_tree_str,
            "ref_tree_str": ref_tree_str,
            "pdm": pairwise_distances,
            "max_dist": max_dist,
            "labels_cont": labels_cont,
            "labels_binary": labels_binary,
            "names": names,
            "corr_coeff_dict": corr_coeff_dict}

def reconstruct_tree(matrix, ids):
    """
    Creates tree from pairwise distance matrix
    Input: (list of [float]) pairwise distance matrix
           (list of str) ids for the matrix
    Output: () reconstructed tree

    From scikit-bio docs: https://scikit.bio/docs/latest/generated/skbio.tree.nj.html#skbio.tree.nj
    """

    # Reconstruct tree using scikit bio
    if matrix.dtype != float:
        matrix = matrix.astype(float)
    dm = DistanceMatrix(matrix, ids)
    tree = nj(dm)
    # tree_str = nj(dm, result_constructor=str)
    tree_str = tree.__str__()

    return tree, dm, tree_str.replace(" ", "")

def rf_distance(tree1_str, tree2_str):
    """
    Calculates Robinson-Foulds distance between two trees
    Input: (str) Newick string of tree 1
           (str) Newick string of tree 2
    Output: (int) output Robinson-Foulds distance
    """

    try:
    
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

    except:
        print("Tree formats are invalid, skipping this sample")
        return {"rf": None,
            "max_rf": None,
            "norm_rf": None}

    return {"rf": rf,
            "max_rf": max_rf,
            "norm_rf": norm_rf}

def mean_cluster_value(names, clusters, labels_cont, labels_binary):

    # Create lookup table for sample labels
    label_dict = {}
    train_labels_cont = []
    for i in range(len(names)):
        label_dict[names[i]] = [labels_cont[i], labels_binary[i]]

    # Create lookup table for sample clusters
    cluster_id = 1
    cluster_dict = {}
    for cluster in clusters:
        if len(cluster) > 1:
            for sample in cluster:
                cluster_dict[sample] = cluster_id
            cluster_id += 1
        else:
            cluster_dict[cluster[0]] = -1

    # Split samples into train/test sets
    names_train, names_test = train_test_split(names, test_size=0.2, random_state=42)
    
    # Save train labels for baseline if test sample is in singleton cluster
    for name in names_train:
        train_labels_cont.append(label_dict[name][0])

    # Calculate predicted value for each test sample
    labels_cont_test = {}
    preds_cont_test = {}
    # Extract continuous labels for test set
    for name in names_test:
        labels_cont_test[name] = label_dict[name][0]
    # Calculate continuous predictions for test set
    for cluster in clusters:
        cluster_label_list = []
        curr_test_names = []
        for name in cluster:
            # Only use train samples for estimate
            if name in names_train:
                cluster_label_list.append(label_dict[name][0])
            elif name in names_test:
                curr_test_names.append(name)
        # Calculate predicted label for test samples 
        if len(cluster_label_list) > 0:
            avg_label = mean(cluster_label_list)
        else: 
            # TODO: If test sample is in singleton cluster, set to average value of training samples in tree
            avg_label = mean(train_labels_cont)
        for name in curr_test_names:
            preds_cont_test[name] = avg_label

    # Calculate spearman rank between labels and preds
    # Sort keys
    sorted_names = sorted(labels_cont_test.keys())
    labels = []
    preds = []
    for name in sorted_names:
        labels.append(labels_cont_test[name])
        preds.append(preds_cont_test[name])
    # Calculate correlation coefficients
    spearman_res = spearmanr(labels, preds)
    pearson_res = pearsonr(labels, preds)
    kendall_res = kendalltau(labels, preds)

    return {"spearman": spearman_res.statistic,
            "pearson": pearson_res.statistic,
            "kendall": kendall_res.statistic}

# Tier 1 metric
def tier1_metric(tree_dicts):

    # Generate Tier 1 evaluation metric for RF distance
    # print("Model: RF Distance")
    tier1_results = {}
    for model_name in tree_dicts.keys():
        curr_tree_dict = tree_dicts[model_name]
        ref_tree_str = curr_tree_dict["ref_tree_str"]
        pred_tree_str = curr_tree_dict["pred_tree_str"]
        # print("%s: %s" % (model_name, rf_distance(ref_tree_str, pred_tree_str)))
        tier1_results[model_name] = rf_distance(ref_tree_str, pred_tree_str)
    return tier1_results

# Tier 2 metric
def tier2_metric(tree_dicts):

    # Generate Tier 2 evaluation metric for average train label value within cluster
    # print("Model: Spearman Rank of Labels vs Preds")
    tier2_results = {}
    for model_name in tree_dicts.keys():
        pred_tree_str = tree_dicts[model_name]["pred_tree_str"]
        names = tree_dicts[model_name]["names"]
        labels_cont = tree_dicts[model_name]["labels_cont"]
        labels_binary = tree_dicts[model_name]["labels_binary"]
        threshold = tree_dicts[model_name]["max_dist"]
        clusters = run_treecluster(pred_tree_str, threshold)
        # print("%s: %s" % (model_name, mean_cluster_value(names, clusters, labels_cont, labels_binary)))
        tier2_results[model_name] = mean_cluster_value(names, clusters, labels_cont, labels_binary)
    return tier2_results

def taxonomic_clustering_benchmark(models, output_file_name, output_file_location, eval_datasets, device, extra_name = ""):
    max_aa_dict = {"ESM2": 19532, "EVO": 62500}
    curr_dir = os.getcwd()
    output_file_path = "%s/%s" % (curr_dir, output_file_name)
    completed_analyses = None

    if os.path.exists(output_file_path):
        completed_analyses = {}
        df = pd.read_csv(output_file_path, sep=",")

        for index, row in df.iterrows():
            taxonomy = row['taxonomy']
            num = row['num']
            if taxonomy not in completed_analyses:
                completed_analyses[taxonomy] = []
            if any(char.isalpha() for char in str(num)):
                num = int(str(num).split('_')[-1])
            completed_analyses[taxonomy].append(num)

    for taxonomy in eval_datasets:
        for num_dataset in range(5):
            if completed_analyses is not None and taxonomy in completed_analyses and num_dataset in completed_analyses[taxonomy]:
                print(f"Already completed {taxonomy} {num_dataset}")
                continue
            else:
                print(f"Starting this taxonomy: {taxonomy} and {num_dataset} analysis")
                sequence = eval_datasets[taxonomy][num_dataset]["sequences"]
                labels = eval_datasets[taxonomy][num_dataset]["labels"]
                if extra_name != "":
                    num_dataset = f'{extra_name}_{num_dataset}'

                # try:
                for model_name in models.keys():

                    labels = pd.read_csv(labels, sep='\t')

                    if os.path.exists(output_file_location+f'{model_name}_{taxonomy}_sequence_embeddings_{num_dataset}.pkl'):
                        print(f"loading premade embeddings for {model_name} on {sequence}")
                        isolate_sequence_embeddings = pickle.load(open(output_file_location+f'{model_name}_{taxonomy}_sequence_embeddings_{num_dataset}.pkl', 'rb'))
                    else:
                        isolate_sequence_embeddings = {}

                        isolate_sequences = pickle.load(open(sequence, "rb"))
                        
                        if "Phyla" in model_name:
                            model, alphabet_tokenizer = models[model_name]["model"], models[model_name]["alphabet_tokenizer"]

                            embeddings = {}
                            for isolate in isolate_sequences:
                                embeddings[isolate] = []

                            num_sequences = 120
                            for num in range(num_sequences):
                                sequences = []
                                names = []

                                for isolate in isolate_sequences:
                                    to_insert = isolate_sequences[isolate][num].replace('-', '')
                                    if len(to_insert) > 0:
                                        sequences.append(isolate_sequences[isolate][num].replace('-', ''))
                                        names.append(isolate)
                                    else:
                                        embeddings[isolate].append(torch.zeros(256))


                                dataset = Arbitrary_Sequence_Dataset()
                                batch, names = dataset.encode_sequences(sequences, names)

                                with torch.no_grad():
                                    sequence_embeddings = model(batch['encoded_sequences'].to(device), 
                                                                batch['sequence_mask'].to(device),
                                                                batch['cls_positions'].bool().to(device))


                                for is_name, embed in zip(names, sequence_embeddings[0].cpu()):
                                    embeddings[is_name].append(embed)

                            for isolate in embeddings:
                                isolate_sequence_embeddings[isolate] = torch.stack(embeddings[isolate])

                        else:
                            model, alphabet_tokenizer = models[model_name]["model"], models[model_name]["alphabet_tokenizer"]
                            #Random bug to check out later keeps going to 0 for some reason
                            model = model.to(device)
                            for isolate in tqdm(isolate_sequences, total = len(isolate_sequences)):
                                sequences = [i.replace('-', '') for i in isolate_sequences[isolate]]
                                
                                if model_name == "ESM2" or model_name == "ESM2_3B":

                                    # Tokenize sequences for input to ESM2
                                    batch_converter = alphabet_tokenizer.get_batch_converter()
                                    data = [(i, sequences[i]) for i in range(len(sequences))]
                                    batch_labels, batch_strs, batch_tokens = batch_converter(data)
                                    batch_lens = (batch_tokens != alphabet_tokenizer.padding_idx).sum(1)

                                    # Generate token representations in batched sequence format
                                    token_representations = torch.Tensor([])
                                    num_seqs = len(batch_lens)
                                    seq_length = batch_tokens.shape[1]
                                    seqs_per_chunk = (max_aa_dict["ESM2"] // seq_length)
                                    token_representations = torch.Tensor([])
                                    
                                    for i in range(0, len(batch_tokens), seqs_per_chunk):
                                        with torch.no_grad():
                                            results = model(batch_tokens[i:i+seqs_per_chunk].to(device), repr_layers=[33], return_contacts=False)
                                        token_representations = torch.cat((token_representations, results["representations"][33].detach().cpu()))

                                    # Generate sequence embeddings
                                    sequence_embeddings = []
                                    for i, tokens_len in enumerate(batch_lens):
                                        sequence_embeddings.append(token_representations[i, 1 : tokens_len - 1].mean(0))
                                    sequence_embeddings = torch.stack(sequence_embeddings).unsqueeze(0)

                                elif model_name == "EVO":
                                    # Convert to the recommended EVO pipeline for generating embeddings
                                    # Convert all protein sequences into nucleotide sequences
                                    # Remove all periods
                                    sequences = [seq.replace(".", "L") for seq in sequences]
                                    sequences = [reverse_translate(seq) for seq in sequences]
                                    # Generate sequence embeddings in batches
                                    logits = torch.Tensor([])
                                    num_seqs = len(sequences)

                                    # For manual padding to deal with different max sizes in sub-batches
                                    max_seq_length = max([len(seq) for seq in sequences])
                                    seqs_per_chunk = (max_aa_dict["EVO"] // max_seq_length)
                                    # device = 'cuda:0'
                                    for i in range(0, num_seqs, seqs_per_chunk):
                                        with torch.no_grad():
                                            input_ids, seq_lengths = prepare_batch(sequences[i:i+seqs_per_chunk], alphabet_tokenizer, prepend_bos=False, device=device)
                                            # Manually pad
                                            curr_num_to_pad = max_seq_length - input_ids.shape[1]
                                            input_ids = torch.cat((input_ids, torch.ones(input_ids.shape[0], curr_num_to_pad, dtype=torch.int64).to(device)),dim=1)
                                            # input_ids = torch.cat((input_ids, torch.ones(input_ids.shape[0], curr_num_to_pad, dtype=torch.int64, device=device)),dim=1)
                                            results, _ = model(input_ids)
                                        # TODO: Logits might not be correct, may need embeddings
                                        logits = torch.cat((logits, results.detach().cpu()))
                                    sequence_embeddings = logits.mean(1).unsqueeze(0)
                                
                                elif model_name == "ESM3":
                                    
                                    # Calculate sequence embeddings 
                                    sequence_embeddings = []
                                    for sequence in sequences:
                                        protein = ESMProtein(sequence=sequence)
                                        protein_tensor = model.encode(protein) 
                                        output = model.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
                                        token_embeddings = output.per_residue_embedding
                                        sequence_embeddings.append(token_embeddings.mean(0).unsqueeze(0))
                                    sequence_embeddings = torch.cat(sequence_embeddings).unsqueeze(0)

                                elif model_name in ["ESMC_300M", "ESMC_600M"]:

                                    # Calculate sequence embeddings 
                                    max_seq_length = max([len(seq) for seq in sequences]) + 2 # Adding +2 because of the built-in ESMC tokenization strategy
                                    sequence_embeddings = []
                                    for sequence in sequences:
                                        protein = ESMProtein(sequence=sequence)
                                        protein_tensor = model.encode(protein)

                                        # Manually pad
                                        curr_num_to_pad = max_seq_length - len(protein_tensor)
                                        protein_tensor.sequence = torch.cat((protein_tensor.sequence, torch.ones(curr_num_to_pad, dtype=torch.int64).to(device)))

                                        logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                                        sequence_embeddings.append((logits_output.embeddings).squeeze().mean(0).unsqueeze(0))
                                    sequence_embeddings = torch.cat(sequence_embeddings).unsqueeze(0)
                                
                                elif model_name == "PROGEN2_XLARGE" or model_name == "PROGEN2_LARGE":
                                    sequence_embeddings = []
                                    for seq in sequences:
                                        if seq == "":
                                            sequence_embeddings.append(torch.zeros(2560).to(device))
                                        else:
                                            token_ids = alphabet_tokenizer.encode(seq).ids
                                            if len(token_ids) > 1024:
                                                chunk_embeddings = []
                                                for i in range(0, len(token_ids), 1024):
                                                    chunk = token_ids[i:i+1024]
                                                    target = torch.tensor(chunk).to(device)
                                                    hidden_states = model(target, labels=target, hidden_state_override=True).detach().cpu()
                                                    chunk_embeddings.append(hidden_states[1:-1].mean(dim=0))  # Exclude BOS and EOS
                                                seq_embedding = torch.stack(chunk_embeddings).mean(dim=0)
                                            else:
                                                target = torch.tensor(token_ids).to(device)
                                                hidden_states = model(target, labels=target, hidden_state_override=True).detach().cpu()
                                                seq_embedding = hidden_states[1:-1].mean(dim=0)  # Exclude BOS and EOS
                                                
                                            sequence_embeddings.append(seq_embedding)
                                
                                elif model_name == "EVO2":
                                    
                                    # sequence_embeddings = []
                                    # for sequence in sequences:
                                    #     input_ids = torch.tensor(alphabet_tokenizer.tokenize(sequence), dtype=torch.int).unsqueeze(0).to('cuda:0')
                                    #     layer_name = 'blocks.28.mlp.l3'
                                    #     outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
                                    #     sequence_embeddings.append(embeddings[layer_name].squeeze().mean(0).unsqueeze(0))
                                    # sequence_embeddings = torch.cat(sequence_embeddings).unsqueeze(0)
                                    # sequence_embeddings = sequence_embeddings.to(torch.float64)

                                    dataset_batch_size = 10

                                    from torch.cuda import OutOfMemoryError
                                    import gc

                                    def get_embeddings(sequences, batch_size):
                                        sequence_embeddings = []
                                        for i in range(0, len(sequences), batch_size):
                                            batch = sequences[i:i + batch_size]
                                            input_ids = [
                                                torch.tensor(alphabet_tokenizer.tokenize(seq), dtype=torch.int)
                                                for seq in batch
                                            ]
                                            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
                                            # input_ids = input_ids.to('cuda:0')

                                            layer_name = 'blocks.28.mlp.l3'
                                            outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
                                            emb = embeddings[layer_name].mean(dim=1)  # mean over sequence length
                                            sequence_embeddings.append(emb)
                                        return torch.cat(sequence_embeddings, dim=0).unsqueeze(0).to(torch.float64)

                                    try:
                                        sequence_embeddings = get_embeddings(sequences, batch_size=dataset_batch_size)
                                    except RuntimeError as e:
                                        if "out of memory" in str(e):
                                            print("OOM at batch_size=2, retrying with batch_size=1")
                                            torch.cuda.empty_cache()
                                            gc.collect()
                                            dataset_batch_size = 1
                                            sequence_embeddings = get_embeddings(sequences, batch_size=dataset_batch_size)
                                        else:
                                            raise e
                                
                                isolate_sequence_embeddings[isolate] = sequence_embeddings
                        
                        if not os.path.exists(output_file_location):
                            os.makedirs(output_file_location)
                        pickle.dump(isolate_sequence_embeddings, open(output_file_location+f'{model_name}_{taxonomy}_sequence_embeddings_{num_dataset}.pkl', 'wb'))
            
                averaged_isolate_sequence_embeddings = {}
                for isolate in isolate_sequence_embeddings:
                    to_process = isolate_sequence_embeddings[isolate]
                    
                    if model_name == "PROGEN2_XLARGE" or model_name == "PROGEN2_LARGE":
                        if model_name == "PROGEN2_XLARGE":
                            dim = 4096
                        else:
                            dim = 2560

                        filtered_to_process = []
                        for i in to_process:
                            if i.shape[0] != dim:
                                filtered_to_process.append(torch.zeros(dim).cpu())
                            else:
                                filtered_to_process.append(i.cpu())

                        to_process = torch.stack(filtered_to_process)

                    if to_process.shape[0] == 1:
                        to_process = to_process.squeeze(0)
                    
                    averaged_isolate_sequence_embeddings[isolate] = torch.nan_to_num(to_process, nan=0.0).mean(axis=1)
                
                id_to_label = labels[['genome_id',taxonomy]]

                # Prepare the data
                embeddings = []
                labels = []

                # Extract embeddings and labels
                for isolate, embedding in averaged_isolate_sequence_embeddings.items():
                    embeddings.append(embedding.cpu().numpy())  # Convert to NumPy array
                    taxonomic_label = id_to_label[id_to_label['genome_id'] == isolate][taxonomy].values[0]
                    labels.append(taxonomic_label)

                # Convert to NumPy arrays
                embeddings = np.vstack(embeddings)  # Shape: (num_samples, embedding_dim)
                labels = np.array(labels)  # Shape: (num_samples,)

                # Perform K-Means clustering
                num_clusters = len(np.unique(labels))  # Number of unique taxonomic labels
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_assignments = kmeans.fit_predict(embeddings)

                # Evaluate cluster homogeneity
                homogeneity = homogeneity_score(labels, cluster_assignments)
                completeness = completeness_score(labels, cluster_assignments)
                v_meas = v_measure_score(labels, cluster_assignments)
                nmi = normalized_mutual_info_score(labels, cluster_assignments)

                output_file_path = "%s/%s" % (curr_dir, output_file_name)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                if not os.path.exists(output_file_path):
                    output_file = open(output_file_path, "w")
                    output_file.write("model,taxonomy,num,homogeneity,completeness,v_measure, nmi\n")
                    output_file.close()

                for model_name in models.keys():
                    output_file = open(output_file_path, "a")
                    output_file.write("%s,%s,%s,%s,%s,%s,%s\n" % (model_name,
                                                            taxonomy,
                                                            num_dataset,
                                                            homogeneity,
                                                            completeness,
                                                            v_meas,
                                                            nmi))
                    
                    print(f"Model: {model_name}, Taxonomy: {taxonomy}, Num: {num_dataset}, Homogeneity: {homogeneity:.4f}, Completeness: {completeness:.4f}, V-measure: {v_meas:.4f}, NMI: {nmi:.4f}")
                    output_file.close()

def functional_prediction_benchmark(models, num_datasets, output_file_name, eval_datasets, device='cuda:0'):

    """
    models: dictionary of loaded models
    num_datasets: list of lowest and highest index of ProteinGym dataset to generate results for
    output_file_name: string output file name
    eval_datasets: list of datasets to perform UMAP evaluation of embeddings
    """
    
    # Load in dataset names sorted from in ascending size
    sequence_paths = []
    tree_paths = []
    curr_dir = os.getcwd()
    summary_file = [dataset.strip().split(",") for dataset in open("%s/eval/proteingym_summary.csv" % curr_dir).readlines()]
    file_names = np.array(summary_file)[:,0][1+num_datasets[0]:num_datasets[1]+1].tolist()
    eval_datasets_names = [summary_file[1+i][-1] for i in eval_datasets]

    # Save spearman values
    linearprobe_spearmans = []
    tier2_spearmans = []

    # Iterate through all datasets
    csv_dir_path = "ProteinGym/"
    temp_tree_path = "" # TODO: Fix after generated all the original trees, now there's no need for reference trees since just doing tier2
    # file_names = file_names[:10]
    for dataset_name in tqdm(file_names, total=len(file_names)):

        # Determine whether to generate UMAPs
        eval_mode = False
        if len(eval_datasets_names) != 0:
            if dataset_name in eval_datasets_names:
                eval_mode = True
            else:
                print("Skipping dataset %s since in eval mode" % dataset_name)
                continue

        # print("Generating results for %s" % dataset_name)

        try: 
            # Generate trees
            sequence_path = "%s/%s.csv" % (csv_dir_path, dataset_name)
            tree_dicts = {}
            time_dict = {}
            for model_name in models.keys():
                
                start_time = time.time()
                tree_dict = generate_tree(seq_file = sequence_path,
                                        tree_file = temp_tree_path,
                                        model = models[model_name]["model"],
                                        alphabet_tokenizer = models[model_name]["alphabet_tokenizer"],
                                        model_name = model_name,
                                        dataset_type = config.dataset.dataset, 
                                        eval_mode = eval_mode, 
                                        device = device)
                end_time = time.time()
                tree_dicts[model_name] = tree_dict
                time_dict[model_name] = end_time - start_time

                if eval_mode:
                    break

                # Save predicted tree into Newick string
                # dir_path = "%s/eval/eval_preds/protein_gym/%s" % (curr_dir, dataset_name)
                # if not os.path.exists(dir_path):
                #     os.mkdir(dir_path)
                # file_path = "%s/%s_pred_tree.nh" % (dir_path, model_name)
                # with open(file_path, "w") as f:
                #     f.write(tree_dict["pred_tree_str"])
                #     f.close()

            # Don't run rest of pipeline if in eval mode
            if eval_mode:
                continue
            
            # Save labels into annotation file (one per dataset)
            # file_path = "%s/annotations.txt" % (dir_path)
            # if not os.path.exists(file_path):
            #     annotation_str = "taxa\tlabels_cont\tlabels_binary\n"
            #     curr_name = tree_dict["names"][0]
            #     curr_label_cont = tree_dict["labels_cont"][0]
            #     curr_label_binary = tree_dict["labels_binary"][0]
            #     annotation_str += "%s\t%s\t%s\n" % (curr_name, curr_label_cont, curr_label_binary)
            #     annotation_str = annotation_str.strip()
            #     with open(file_path, "w") as f:
            #         f.write(annotation_str)
            #         f.close()

            # Evaluate Tier 2
            tier2_dict = tier2_metric(tree_dicts)

            # Keep track of entire runtime for each dataset
            end_time_full = time.time()
            elapsed_time_full = end_time_full - start_time

            # Write results to output file
            linearprobe_spearmans.append(tree_dicts[model_name]["corr_coeff_dict"]["spearman"])
            tier2_spearmans.append(tier2_dict[model_name]["spearman"])
            output_file_path = "%s/%s" % (curr_dir, output_file_name)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True) 
            if not os.path.exists(output_file_path):
                output_file = open(output_file_path, "w")
                output_file.write("dataset,model,tier2_spearman,tier2_pearson,tier2_kendall,linear_probe_spearman,linear_probe_pearson,linear_probe_kendall,runtime,full_runtime\n")
                # output_file.write("dataset,model,linear_probe_spearman,linear_probe_pearson,linear_probe_kendall,runtime,full_runtime\n")
                output_file.close()
            for model_name in models.keys():
                output_file = open(output_file_path, "a")
                # output_file.write("%s,%s,%s,%s,%s,%s,%s\n" % (dataset_name,
                output_file.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (dataset_name,
                                                        model_name,
                                                        tier2_dict[model_name]["spearman"],
                                                        tier2_dict[model_name]["pearson"],
                                                        tier2_dict[model_name]["kendall"],
                                                        tree_dicts[model_name]["corr_coeff_dict"]["spearman"],
                                                        tree_dicts[model_name]["corr_coeff_dict"]["pearson"],
                                                        tree_dicts[model_name]["corr_coeff_dict"]["kendall"],
                                                        time_dict[model_name],
                                                        elapsed_time_full))
                output_file.close()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("OOM for dataset %s" % dataset_name)
            else:
                raise e
    
    output_model_name = output_file_name.split("results_")[-1][:-4]
    print(f"\nAverage Linear Probe Spearman for {output_model_name}: {sum(linearprobe_spearmans)/len(linearprobe_spearmans)}")
    print(f"\nAverage Tier 2 Spearman for {output_model_name}: {sum(tier2_spearmans)/len(tier2_spearmans)}")
    print(f"Number of Successes for {output_model_name}: {len(tier2_spearmans)}")

def tree_reconstruction_benchmark(models, num_datasets, output_file_name, dataset_type, dictionary_data = None, device = 'cuda:0'):

    """
    models: dictionary of loaded models
    num_datasets: list of lowest and highest index of Openfold dataset to generate results for
    output_file_name: string output file name
    dataset_type: type of openfold dataset to evaluate between "openfold" and "openfold_small"
    """
    
    # Load in dataset names sorted from in ascending size
    curr_dir = os.getcwd()
    if dataset_type == "treebase":
        # file_names = np.array(os.listdir(""))
        file_names = np.loadtxt("./data/treebase_datasets_1533.txt", dtype=str)
        file_names = file_names[num_datasets[0]:num_datasets[1]]
    elif (dictionary_data is not None or output_file_name == None) and dataset_type == "treefam":
        if output_file_name == None:
            dictionary_data = pickle.load(open("", 'rb'))
            file_names = list(dictionary_data.keys())[:500]
        else:
            file_names = list(dictionary_data.keys())
    # Iterate through all datasets
    if dataset_type == "openfold":
        seq_dir_path = ""
    elif dataset_type == "openfold_small":
        seq_dir_path = ""
    elif dataset_type == "openfold_3tree":
        seq_dir_path = ""
    elif dataset_type == "treebase":
        # seq_dir_path = ""
        seq_dir_path = "./treebase_benchmark/sequences/"
    elif dataset_type == "treefam":
        seq_dir_path = ""
    # Define paths to tree directories
    if dataset_type == "treebase":
        # tree_dir_path = ""
        tree_dir_path = "./treebase_benchmark/trees/"
    elif dataset_type == "treefam":
        tree_dir_path = ""
    else:
        tree_dir_path = ""   # Using the approximated trees generated

    # Iterate
    normrfs = [] # For saving normRF values if running validation callback
    tree_sizes = []

    output_file_path = "%s/%s" % (curr_dir, output_file_name)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    output_file = open(output_file_path, "w")

    # Iterate through all datasets
    for dataset_name in tqdm(file_names, total=len(file_names)):

        # Skipping one dataset that triggers custom error
        if dataset_name == "TF352211":
            print("Skipping dataset %s" % dataset_name)
            continue

        try: 
            # Keep track of entire runtime for each dataset
            start_time_full = time.time()

            # Generate trees
            sequence_path = "%s/%s.fa" % (seq_dir_path, dataset_name)
            tree_path = "%s/%s_tree.nh" % (tree_dir_path, dataset_name)
            tree_dicts = {}
            time_dict = {}
            general_error = False
            for model_name in models.keys():

                tree_dict = generate_tree(seq_file = sequence_path,
                                            tree_file = tree_path,
                                            model = models[model_name]["model"],
                                            alphabet_tokenizer = models[model_name]["alphabet_tokenizer"],
                                            model_name = model_name,
                                            dataset_type = dataset_type, 
                                            eval_mode = False,
                                            convert_to_aa = False,
                                            dictionary_data = dictionary_data[dataset_name] if dictionary_data is not None else None,
                                            random = 42,
                                            device = device
                                            )
                
                start_time = time.time()
                # TODO: Added in temporary try-catch for ESM2 and ESM2_3B for TreeBase
                # try:
                #     tree_dict = generate_tree(seq_file = sequence_path,
                #                             tree_file = tree_path,
                #                             model = models[model_name]["model"],
                #                             alphabet_tokenizer = models[model_name]["alphabet_tokenizer"],
                #                             model_name = model_name,
                #                             dataset_type = config.dataset.dataset, 
                #                             eval_mode = False)
                # except:
                #     print("General unknown error, skipping this sample")
                #     general_error = True
                #     continue

                end_time = time.time()
                tree_dicts[model_name] = tree_dict
                time_dict[model_name] = end_time - start_time

                # Save predicted tree into Newick string
                # if dataset_type == "openfold":
                #     dir_path = "%s/eval/eval_preds/openfold/%s" % (curr_dir, dataset_name)
                # elif dataset_type == "openfold_small":
                #     dir_path = "%s/eval/eval_preds/openfold_small/%s" % (curr_dir, dataset_name)
                # elif dataset_type == "openfold_3tree":
                #     dir_path = "%s/eval/eval_preds/openfold_3tree/%s" % (curr_dir, dataset_name)
                # if not os.path.exists(dir_path):
                #     os.mkdir(dir_path)
                # file_path = "%s/%s_pred_tree.nh" % (dir_path, model_name)
                # with open(file_path, "w") as f:
                #     f.write(tree_dict["pred_tree_str"])
                #     f.close()

            # TODO: Added in temporary try-catch for ESM2 and ESM2_3B for TreeBase
            if general_error:
                continue

            # Evaluate Tier 1
            tier1_dict = tier1_metric(tree_dicts)

            # Keep track of entire runtime for each dataset
            end_time_full = time.time()
            elapsed_time_full = end_time_full - start_time

            # If running code in validation callback, skip writing to output file
            if output_file_name == None:
                normrfs.append(tier1_dict["PHYLA"]["norm_rf"])
                tree_sizes.append(len(tree_dict['names']))
            # If not running code in validation callback, write to output file
            else:
                # Still add values to normrfs 
                # model_id = output_file_name.split("_")[-1].replace('.csv', '')
                model_id = output_file_name.split("_")[-2]
                if '-' in model_id:
                    model_id = model_id.split('-')[0]

                normrfs.append(tier1_dict[model_id]["norm_rf"])
                # Write results to output file
                output_file_path = "%s/%s" % (curr_dir, output_file_name)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                if not os.path.exists(output_file_path):
                    output_file = open(output_file_path, "w")
                    output_file.write("dataset,model,rf,max_rf,norm_rf,runtime,full_runtime\n")
                    output_file.close()
                for model_name in models.keys():
                    output_file = open(output_file_path, "a")
                    output_file.write("%s,%s,%s,%s,%s,%s,%s\n" % (dataset_name,
                                                            model_name,
                                                            tier1_dict[model_name]["rf"],
                                                            tier1_dict[model_name]["max_rf"],
                                                            tier1_dict[model_name]["norm_rf"],
                                                            time_dict[model_name],
                                                            elapsed_time_full))
                    output_file.close()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("OOM for dataset %s" % dataset_name)
            else:
                raise e
    
            
    # If running code in validation callback, return all predicted normRF values
    if output_file_name == None:
        # return normrfs
        return normrfs, tree_sizes
    else:
        output_model_name = output_file_name.split("results_")[-1][:-4]
        print(f"\nAverage NormRF for {output_model_name}: {sum(normrfs)/len(normrfs)}")
        print(f"Number of Successes for {output_model_name}: {len(normrfs)}")


if __name__ == "__main__":
    print("\nRunning evaluate...")
    config = load_config(Config)

    # Load model
    print("\nLoading model %s..." % config.trainer.model_type)
    models = {}
    if "Phyla" in config.trainer.model_type:
        phyla_model_dict = load_model(checkpoint_file = config.trainer.checkpoint_path, config = config.model, device = config.eval.device)
        models["Phyla"] = phyla_model_dict
    elif config.trainer.model_type == "ESM2":
        esm2_model_dict = load_model(config=ESM2_ModelConfig())
        models["ESM2"] = esm2_model_dict
    elif config.trainer.model_type == "EVO":
        evo_model_dict = load_model(config=EVO_ModelConfig())
        models["EVO"] = evo_model_dict
    elif config.trainer.model_type == "ESM3":
        # Only load ESM3 libraries if using ESM3
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, SamplingConfig
        from esm.utils.constants.models import ESM3_OPEN_SMALL
        esm3_model_dict = load_model(config=ESM3_ModelConfig())
        models["ESM3"] = esm3_model_dict
    elif config.trainer.model_type == "ESM2_3B":
        esm2_3b_model_dict = load_model(config=ESM2_3B_ModelConfig())
        models["ESM2_3B"] = esm2_3b_model_dict

    # Generate Tier1 or Tier2 results
    if config.dataset.dataset == "protein_gym":
        if 'ProteinGym' not in os.listdir():
            os.system("wget https://tinyurl.com/mvudt6fd")
            os.system("tar -xvf mvudt6fd")
            os.system("rm mvudt6fd")

        last_dataset_id = 83
        output_file = "eval/eval_preds/protein_gym/protein_gym_results_%s.csv" % (config.trainer.model_type)
        num_datasets = [0, last_dataset_id]   # Start with the smallest 83 datasets for MAMBA's current 80GB GPU constraints
        eval_datasets = []
        functional_prediction_benchmark(models, num_datasets, output_file, eval_datasets, device = config.eval.device)
    
    elif config.dataset.dataset == "treefam":
        if 'treefam.pickle' not in os.listdir():
            os.system("wget https://tinyurl.com/yh78swxd")
            os.system("mv yh78swxd treefam.pickle")

        data = pickle.load(open("treefam.pickle", 'rb'))
        num_datasets = [0, len(data)]
        last_dataset_id = len(data)

        output_file = "eval/eval_preds/treefam/treefam_results_%s_%s.csv" % (config.trainer.model_type, config.eval.extra_name)

        tree_reconstruction_benchmark(models, num_datasets, output_file, config.dataset.dataset, dictionary_data=data, device = config.eval.device)

    elif config.dataset.dataset == "treebase":
        if 'treebase_benchmark' not in os.listdir():
            os.system("wget https://tinyurl.com/ke8pjyw7")
            os.system("unzip ke8pjyw7")
            os.system("rm ke8pjyw7")

        last_dataset_id = 5822
        output_file = "eval/eval_preds/treebase/treebase_results_%s.csv" % (config.trainer.model_type)
        num_datasets = [0, last_dataset_id]
        tree_reconstruction_benchmark(models, num_datasets, output_file, config.dataset.dataset, device = config.eval.device)
    
    elif config.dataset.dataset == "GTB":
        if '3rhyfu7t' not in os.listdir():
            os.system("wget https://tinyurl.com/3rhyfu7t")
            os.system("tar -xvf 3rhyfu7t")
            os.system("rm 3rhyfu7t")

        last_dataset_id = 120

        output_file = "eval/eval_preds/GTB/GTB_results_%s_%s.csv" % (config.trainer.model_type, config.eval.extra_name)

        num_datasets = [0, last_dataset_id]

        taxonomy_to_files = {}
        location = "GTDB_taxonomic_clustering/"
        for filename in os.listdir(location):
            if 'sampled_bac120_taxonomy' in filename:
                if '.pickle' in filename:
                    taxonomy = filename.split("_")[-3]
                    num = int(filename.split("_")[-1].split('.')[0])
                    if taxonomy not in taxonomy_to_files:
                        taxonomy_to_files[taxonomy] = {}
                    if num not in taxonomy_to_files[taxonomy]:
                        taxonomy_to_files[taxonomy][num] = {}
                    taxonomy_to_files[taxonomy][num]['sequences'] = f'{location}{filename}'
                elif '.tsv' in filename:
                    taxonomy = filename.split("_")[-2]
                    num = int(filename.split("_")[-1].split('.')[0])
                    if taxonomy not in taxonomy_to_files:
                        taxonomy_to_files[taxonomy] = {}
                    if num not in taxonomy_to_files[taxonomy]:
                        taxonomy_to_files[taxonomy][num] = {}
                    taxonomy_to_files[taxonomy][num]['labels'] = f'{location}{filename}'

        taxonomic_clustering_benchmark(models, output_file, "eval/eval_preds/GTB/", taxonomy_to_files, device = config.eval.device )
    else:
        raise Exception("You need a dataset name")
