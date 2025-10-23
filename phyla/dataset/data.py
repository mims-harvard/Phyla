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
