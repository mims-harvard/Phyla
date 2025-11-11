# Import packages
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import random
import numpy as np
import os
from Bio import Phylo
from torch.utils.data import Sampler, DistributedSampler
from os.path import exists
from tqdm import tqdm
import pickle
import logging

class Arbitrary_Sequence_Dataset(pl.LightningDataModule):
    #Assume for now we can fit all into memory
    def __init__(self):
        super().__init__()
        self.amino_acid_list = ["A", "R", "N", "D", "C",
                                "Q", "E", "G", "H", "I",
                                "L", "K", "M", "F", "P",
                                "S", "T", "W", "Y", "V"]
        self.amino_acid_encoding = {"A": 1, "R": 2, "N": 3, "D": 4, "C": 5,
                                   "Q": 6, "E": 7, "G": 8, "H": 9, "I": 10,
                                   "L": 11, "K": 12, "M": 13, "F": 14, "P": 15,
                                   "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20}
        #21 is mask, 22 is CLS, 23 is PAD

    def read_tree(self, tree_file):
        with open(tree_file, 'r') as file:
            tree = file.read().strip()
        return tree

    def read_distance(self, distance_file):
        with open(distance_file, 'rb') as file:
            distance_matrix = pickle.load(file)
        return distance_matrix

    def encode(self, sequence):
        """
        Performs integer encoding for each amino acid in input protein sequence
        Input: (str) amino acid sequence
        Output: (list of [float]) integer encoded representation of protein sequence
        """
        # Initialize variables
        sequence_encoded = []

        # Iterate through all amino acids in sequence
        for i in range(len(sequence)):
            curr_amino_acid = sequence[i]
            if curr_amino_acid not in self.amino_acid_encoding.keys():
                sequence_encoded.append(23)
            else:
                sequence_encoded.append(self.amino_acid_encoding[curr_amino_acid])
        
        return sequence_encoded

    def mask(self, sequence):
        """
        Perform masking on encoded input sequence for masked language task
        Input: (list of int) Encoded input sequence
        Output: (list of int) Masked encoded output sequence
                (list of int) Positions of masked amino acids
                (list of int) Identities of encoded masked amino acids

        Note: value of 21 is equivalent to the masked amino acid
        """
        # Initialize variables
        masked_sequence = []
        masked_positions = [0]*len(sequence)
        masked_identities = sequence

        # Iterate through each encoded amino acid
        for i in range(len(sequence)):
            # Check whether to mask amino acid
            if random.random() < 0.15:
                # If so, then update variables
                masked_sequence.append(21) # Represents masked amino acid
                masked_positions[i] = 1
            else:
                masked_sequence.append(sequence[i])
        
        return masked_sequence, masked_positions, masked_identities
    
    def encode_sequences(self, sequences, names, randomize_order = False):
        # Access tree with name
        final_true_seq = []

        if randomize_order:
            # Randomize the order of the sequences
            paired_data = list(zip(sequences, names))
            random.Random(randomize_order).shuffle(paired_data)
            sequences, names = zip(*paired_data)

        # Iterate through all protein sequences in tree
        for seq in sequences:

            # Encode and mask sequence
            encoded_seq = self.encode(seq)

            final_true_seq.append(encoded_seq)

        # return self.collate_fn([[final_true_seq]])
        return self.collate_fn([[final_true_seq]]), names


    def collate_fn(self, batch):

        # Initialize variables
        final_batch = {}

        cls_position = []
        encoded_sequences = []
        sequence_mask = []
        sequence_lengths = []

        for tree in batch:
                # Calculate longest sequence
                combined_sequence = []
                combined_cls_position = []
                combined_sequence_mask = []
                combined_sequence_lengths = []

                #Flatening this for input to momba
                for i in range(len(tree[0])):
                    combined_sequence.append(22)
                    combined_cls_position.append(1)
                    combined_sequence.extend(tree[0][i])
                    combined_cls_position.extend([0]*len(tree[0][i]))
                    combined_sequence_mask.extend([i]*(len(tree[0][i])+1))
                    combined_sequence_lengths.append(len(tree[0][i])+1)

                encoded_sequences.append(combined_sequence)
                cls_position.append(combined_cls_position)
                sequence_mask.append(combined_sequence_mask)
                sequence_lengths.append(combined_sequence_lengths)


        final_batch["encoded_sequences"] = torch.IntTensor(encoded_sequences)
        final_batch["cls_positions"] = torch.IntTensor(cls_position)
        final_batch['sequence_mask'] = torch.IntTensor(sequence_mask)
        final_batch["sequence_lengths"] = torch.IntTensor(sequence_lengths)

        return final_batch

class SizeDetector():
    #Returns max sub-tree size you can fit on current GPU based on current sequence max length
    def __init__(self):
        self.gpu_memory = self.gpu_memory_detector()

        self.gpu_max_aa = {32: 21000,
                            48 : 30000,
                            80: 41000}

        self.memory_model = np.polyfit(list(self.gpu_max_aa.keys()), list(self.gpu_max_aa.values()), 1)
        self.max_aa = np.polyval(self.memory_model, self.gpu_memory)

        print(f"Detected that have {self.gpu_memory} GB of GPU memory for max AA of {self.max_aa}")
    
    def gpu_memory_detector(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
        else:
            return "No GPU available"
    
    def update_max_aa(self, max_aa):
        if max_aa < self.max_aa:
            self.max_aa = max_aa
            print(f"Detected inaccurate max aa estimate updating max AA to {self.max_aa}")

    def return_subtree_size(self, max_seq_length):
        self.max_seq_length = max_seq_length
        return int(self.max_aa/max_seq_length)

    def return_number_subtree(self, sub_tree_size):
        tree_aa = sub_tree_size * self.max_seq_length
        return int(self.max_aa/tree_aa)

class OpenFold_Dataset(pl.LightningDataModule):
    def __init__(self, dataset_directories, 
                 logger, 
                 dataset_size = None):
        
        super().__init__()
        self.dataset_directories = dataset_directories
        self.dataset_size = dataset_size
        self.size_detector = SizeDetector()

        self.minimum_tree = 10

        self.amino_acid_list = ["A", "R", "N", "D", "C",
                                "Q", "E", "G", "H", "I",
                                "L", "K", "M", "F", "P",
                                "S", "T", "W", "Y", "V"]
        
        self.nucleotide_list = ["A", "C", "G", "T", "U"]

        self.amino_acid_encoding = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
                                   "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
                                   "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
                                   "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19, 'X':20}

        self.nucleotide_encoding = {"A": 21, "C": 22, "G": 23, "T": 24, "U": 25, 'Y': 26}

        self.mask_token = 27
        self.cls_token = 28
        self.pad_token = 29

        #27 is mask, 28 is CLS, 29 is PAD
        self.tree_map = self.gather_data()
        print(len(self.tree_map))
        self.logger = logger
        self.current_tree_size = 5 

    def gather_data(self):
        file_mapping = {}

        num_completed = 0
        for directory in self.dataset_directories:
            for i in tqdm(os.listdir(directory)):
                if '.fasta' in i or '.npy' in i:
                    name = i.split('_')[0]
                    if name in file_mapping:
                        file_mapping[name].append(f'{directory}/{i}')
                        num_completed += 1
                    else:
                        file_mapping[name] = [f'{directory}/{i}']

                if self.dataset_size is not None and num_completed == self.dataset_size:
                    to_return = {}
                    for key in file_mapping:
                        if len(file_mapping[key]) == 2:
                            to_return[key] = file_mapping[key]  
                    return to_return

        #Filter for incomplete data
        to_return = {}
        for key in file_mapping:
            if len(file_mapping[key]) == 2:
                to_return[key] = file_mapping[key]  
        return to_return

    def read_sequences(self, fasta_file, return_ordering = False):
        name_to_seq = {}
        seq_ordering = []
        for line in open(fasta_file, 'r').readlines():
            if '>' in line:
                name = line.strip()
                name_to_seq[name] = ""
                seq_ordering.append(name)
            else:
                name_to_seq[name] += line.strip().replace('-', '').upper()

        if return_ordering:
            return name_to_seq, seq_ordering
        return name_to_seq
    
    def return_max_length(self, name_to_seq):
        return max([len(i) for i in name_to_seq.values()])

    def read_tree(self, tree_file):
        with open(tree_file, 'r') as file:
            tree = file.read().strip()
        return tree

    def read_distance(self, distance_file, sequence_ordering):
        if '.npy' in distance_file:
            return np.load(distance_file)
        elif '.csv' in distance_file:
            df = pd.read_csv(distance_file, header=None)

            df.columns = df.iloc[0]
            df = df[1:]
            df.index = df.iloc[:, 0]
            df = df.iloc[:, 1:]

            def extract_clean_id(name):
                data = name.split('|')
                return (data[1] + data[2]).split(' ')[0]

            ordered_ids = [extract_clean_id(name) for name in self.seq_ordering]
 
            df = df.loc[ordered_ids, ordered_ids] 
            x = df.values.astype(float)
            x = x/x.max()
            #For really dumb reasons we do this so it cancels out the 1-
            return 1 - x
        with open(distance_file, 'rb') as file:
            distance_matrix = pickle.load(file)
        return distance_matrix
    
    def __len__(self):
        return len(self.tree_map)

    def encode(self, sequence):
        """
        Performs integer encoding for each amino acid in input protein sequence
        Input: (str) amino acid sequence
        Output: (list of [float]) integer encoded representation of protein sequence
        """
        # Initialize variables
        sequence_encoded = []

        present_characters = set(sequence)
        num_in_AA = len(present_characters.intersection(set(self.amino_acid_list)))
        if num_in_AA / len(present_characters) >= 0.75:
            encoding_dict = self.amino_acid_encoding
            junk_token = encoding_dict['X']
        else:
            encoding_dict = self.nucleotide_encoding
            junk_token = encoding_dict['Y']

        # Iterate through all amino acids in sequence
        for i in range(len(sequence)):
            curr_amino_acid = sequence[i]
            if curr_amino_acid not in encoding_dict.keys():
                sequence_encoded.append(junk_token)
            else:
                sequence_encoded.append(encoding_dict[curr_amino_acid])

        return sequence_encoded

    def mask(self, sequence):
        """
        Perform masking on encoded input sequence for masked language task
        Input: (list of int) Encoded input sequence
        Output: (list of int) Masked encoded output sequence
                (list of int) Positions of masked amino acids
                (list of int) Identities of encoded masked amino acids

        Note: value of 21 is equivalent to the masked amino acid
        """
        # Initialize variables
        masked_sequence = []
        masked_positions = [0]*len(sequence)
        masked_identities = sequence

        # Iterate through each encoded amino acid
        for i in range(len(sequence)):
            # Check whether to mask amino acid and don't mask unknown amino acids
            if random.random() < 0.15 and sequence[i] != 23:
                # If so, then update variables
                masked_sequence.append(21) # Represents masked amino acid
                masked_positions[i] = 1
            else:
                masked_sequence.append(sequence[i])
        
        return masked_sequence, masked_positions, masked_identities
    
    def __getitem__(self, index, preset_subtree_size = None):
        """
        Get single tree and format each sequence in tree into masked encoded chunks of length 512
        Input: (int) Name of tree to access in self.dataset
        Output: (list of dims [3, 10, 3, 512]) Output tree with chunked encoded masked sequences with items:
                    (list of dims [10, 3, 512]) masked sequences
                    (list of dims [10, 3, 512]) masked positions
                    (list of dims [10, 3, 512]) masked identities 
        """
        # Access tree with name
        #Fix for distance matrix not being able to be loaded
        valid_files = False
        while not valid_files:
            try:
                file_one, file_two = self.tree_map[index]
                if '.fasta' in file_one:
                    fasta_path = file_one
                    distance_path = file_two 
                else:
                    fasta_path = file_two
                    distance_path = file_one
                
                self.name_to_seq, self.seq_ordering = self.read_sequences(fasta_path, return_ordering=True)
                distance_matrix = 1-self.read_distance(distance_path, self.seq_ordering)
                # valid_files = True
                if len(self.name_to_seq) < 10:
                    print("TOO SMALL of a tree, skipping")
                    keys = list(self.tree_map.keys())
                    index = random.choice(keys)
                    valid_files = False
                else:
                    valid_files = True
            except:
                #choose random key in tree_map
                keys = list(self.tree_map.keys())
                index = random.choice(keys)
                self.logger.log(f"Distance file {distance_path} was not able to be loaded, choose new index, {index}", level=logging.WARNING)
                #print(f"Distance file {distance_path} was not able to be loaded, choose new index, {index}")

        final_masked_tree = []
        final_masked_pos = []
        final_masked_id = []
        final_true_seq = []

        if preset_subtree_size:
            sub_tree_size = preset_subtree_size
            self.logger.log(f"Detected preset subtree size {sub_tree_size}, pulling", level=logging.INFO)
            self.chosen_tree = [index, sub_tree_size, self.chosen_tree[2]]

        else:
            max_sub_tree_size = self.size_detector.return_subtree_size(self.return_max_length(self.name_to_seq))
            # For larger model
            num_sequences = len(self.name_to_seq)
            if num_sequences <= self.minimum_tree:
                sub_tree_size = num_sequences
            else:
                if min(max_sub_tree_size, num_sequences) < self.minimum_tree:
                    sub_tree_size = self.minimum_tree
                else:
                    sub_tree_size = random.randint(self.minimum_tree, min(max_sub_tree_size, num_sequences))
                    #If you uncomment this you kill adaptive batch size
                    # if sub_tree_size > self.current_tree_size:
                    #     sub_tree_size = self.current_tree_size
            
            if torch.cuda.device_count() > 1: 
                if sub_tree_size > 100:
                    sub_tree_size = self.current_tree_size


            self.chosen_tree = [index, sub_tree_size, None]

            self.logger.log(f"Adaptive batch size dictates {sub_tree_size} sequences can fit on GPU, chose between {5} and {min(max_sub_tree_size, num_sequences)}", level=logging.INFO)

        # Perform sub-sampling of larger tree
        subtree_sequences = [i for i in random.sample(list(self.name_to_seq.keys()), sub_tree_size)] 
        sequences = self.seq_ordering

        # Initialize the distance matrix
        dm = np.zeros((len(subtree_sequences), len(subtree_sequences)))

        for i in range(len(subtree_sequences)):
            for j in range(i, len(subtree_sequences)):
                seq_i = subtree_sequences[i]
                seq_j = subtree_sequences[j]
                dm[i][j] = distance_matrix[sequences.index(seq_i), sequences.index(seq_j)]
                dm[j][i] = dm[i][j]
                
        # Iterate through all protein sequences in tree
        for seq_name in subtree_sequences:
            seq = self.name_to_seq[seq_name]

            # Encode and mask sequence
            encoded_seq = self.encode(seq)
            masked_seq, masked_pos, masked_id = self.mask(encoded_seq)

            final_masked_tree.append(masked_seq)
            final_masked_pos.append(masked_pos)
            final_masked_id.append(masked_id)
            final_true_seq.append(encoded_seq)

        to_return = [final_masked_tree, final_masked_pos, final_masked_id, dm, 
            [i.replace('>', '').split(' ')[0] for i in subtree_sequences], final_true_seq, index]

        return to_return

    def collate_fn(self, batch, preset_subtree_num = None):
        """
        Returns batched format for each batch in dataset
        Input: (list of dims [1, 3, 10, 3, 512]) current batch
        Output: (dict) Formatted current batch with key-value pairs:
                    (str) masked_sequences : (int tensor of dims [1, 10, 3, 512]) masked features
                    (str) masked_positions : (int tensor of dims [1, 10, 3, 512]) masked positions
                    (str) masked_identities : (int tensor of dims [1, 10, 3, 512]) masked identities
                    (str) tree_matrix : (float tensor of dims [1, 10, 10]) pairwise distance matrix
                    (str) tree_labels : (list of dims [1, 10]) labels for tree
                    
        Note: batch dimensions are [batch_size, 3, num_sequences, num_chunks, length_chunk]
              current batch dimensions is [1, 3, 10, 3, 512]
        Note: the second dimension is 3 because __getitem__ returns items of masked sequences, masked positions, and masked identities
        """

        # Initialize variables
        final_batch = {}
        masked_sequences = []
        masked_positions = []
        masked_identities = []
        padded_positions = []
        tree_matrices = []
        tree_labels = []
        sequence_positions = []

        cls_position = []
        true_sequences = []
        sequence_mask = []

        if preset_subtree_num:
            tree, sub_tree_size, _ = self.chosen_tree
            number_sub_tree = preset_subtree_num
            # TODO: Also need to remove the cap on the number of subtrees
            if number_sub_tree > 100:
                number_sub_tree = 50
            self.chosen_tree[2] = number_sub_tree
        else:
            tree, sub_tree_size, _ = self.chosen_tree
            number_sub_tree = self.size_detector.return_number_subtree(sub_tree_size)
            # TODO: Also need to remove the cap on the number of subtrees
            if number_sub_tree > 100:
                number_sub_tree = 50
            self.chosen_tree[2] = number_sub_tree

        if number_sub_tree > 1:
            self.logger.log(f"Adaptive batch size dictates {number_sub_tree} subtrees can fit on GPU", level=logging.INFO)
            for i in range(1, number_sub_tree):
                batch.append(self.__getitem__(tree, preset_subtree_size = sub_tree_size))

        tree_indices = []
        for tree in batch:
                # Calculate longest sequence
                longest_subsequence_length = max([len(i) for i in tree[0]])
                combined_sequence = []
                combined_positions = []
                combined_masked_identities = []
                combined_cls_position = []
                combined_true_sequence = []
                combined_padded_positions = []
                combined_sequence_mask = []

                #Flatening this for input to momba
                for i in range(len(tree[0])):
                    combined_sequence.append(self.cls_token)
                    combined_cls_position.append(1)
                    combined_sequence.extend(tree[0][i])
                    combined_cls_position.extend([0]*len(tree[0][i]))

                    combined_true_sequence.append(self.cls_token)
                    combined_true_sequence.extend(tree[5][i])

                    combined_positions.append(0)
                    combined_positions.extend(tree[1][i])

                    combined_masked_identities.append(self.cls_token)
                    combined_masked_identities.extend(tree[2][i])

                    # Note padded positions
                    combined_padded_positions.append(0)
                    combined_padded_positions.extend([0]*len(tree[1][i]))

                    combined_sequence_mask.extend([i]*(len(tree[0][i])+1))

                masked_sequences.append(combined_sequence)
                masked_positions.append(combined_positions)
                masked_identities.append(combined_masked_identities)
                padded_positions.append(combined_padded_positions)
                cls_position.append(combined_cls_position)
                tree_matrices.append(tree[3])
                tree_labels.append(tree[4])
                true_sequences.append(combined_true_sequence)
                sequence_mask.append(combined_sequence_mask)
                tree_indices.append(tree[6])

        longest_sequence_length = max([len(i) for i in masked_sequences])
        padded_positions = []
        for i in range(len(masked_sequences)):
            masked_sequences[i].extend([self.pad_token]*(longest_sequence_length - len(masked_sequences[i])))
            masked_positions[i].extend([0]*(longest_sequence_length - len(masked_positions[i])))
            masked_identities[i].extend([self.pad_token]*(longest_sequence_length - len(masked_identities[i])))
            true_sequences[i].extend([self.pad_token]*(longest_sequence_length - len(true_sequences[i])))
            cls_position[i].extend([0]*(longest_sequence_length - len(cls_position[i])))
            padded_positions.append([0]*len(masked_positions[i])+[1]*(longest_sequence_length - len(masked_positions[i])))
            sequence_mask[i].extend([-1]*(longest_sequence_length - len(sequence_mask[i])))

        final_batch["masked_sequences"] = torch.IntTensor(masked_sequences)
        final_batch["masked_positions"] = torch.IntTensor(masked_positions)
        final_batch["masked_identities"] = torch.IntTensor(masked_identities)
        final_batch["padded_positions"] = torch.IntTensor(padded_positions)
        final_batch["cls_positions"] = torch.IntTensor(cls_position)
        final_batch["tree_matrix"] = torch.FloatTensor(tree_matrices)
        final_batch["tree_labels"] = tree_labels
        final_batch["true_sequences"] = torch.IntTensor(true_sequences)
        final_batch["sequence_positions"] = sequence_positions
        final_batch['sequence_mask'] = torch.IntTensor(sequence_mask)
        
        final_tree_index = list(set(tree_indices))
        if len(final_tree_index) > 1:
            raise Exception("It should not be possible that one batch is from multiple trees")
        flat_tree_labels = []
        for i in tree_labels:
            for j in i:
                if j not in flat_tree_labels:
                    flat_tree_labels.append(j)
                    
        return final_batch

    def train_dataloader(self):
        return DataLoader(self,
                            num_workers = 0,
                            batch_size=1,
                            collate_fn=self.collate_fn,
                            sampler  = OpenFold_TreeSampler(self.dataset_directories, self.dataset_size))

    def val_dataloader(self):
        # Define a dummy dataset with one batch
        class DummyDataset(Dataset):
            def __len__(self):
                return 1  # Single batch
            def __getitem__(self, index):
                return {"dummy_input": torch.tensor([0])}  # Dummy data

        return DataLoader(DummyDataset(), batch_size=1)

class OpenFold_TreeSampler(DistributedSampler):
    def __init__(self, dataset_directory, dataset_size):
        self.trees = []

        for directory in dataset_directory:
            for i in os.listdir(directory):
                name = i.split('_')[0]
                if name not in self.trees:
                    self.trees.append(name)

                # Check if have pulled up to the dataset_size
                if len(self.trees) == dataset_size:
                    return

    def __len__(self):
        return len(self.trees)

    def __iter__(self):
        while True:
            yield random.choice(self.trees)
