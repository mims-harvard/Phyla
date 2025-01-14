# Phyla: Towards a Foundation Model for Phylogenetic Inference

![Tree of life](img/16S_sequences.png)

## What is Phyla? 

Phyla is a protein language model designed to model both individual sequences and inter-sequence relationships. It leverages a hybrid state-space transformer architecture and is trained on two tasks: masked language modeling and phylogenetic tree reconstruction using sequence embeddings. Phyla enables rapid construction of phylogenetic trees of protein sequences, offering insights that differ from classical methods in potentially functionally significant ways.

## Disclaimer

We are excited to introduce Phyla-α, an early-stage version of our model that is still under active development. Future iterations will incorporate methodological improvements and additional training data as we continue refining the model. Please note that this work is ongoing, and updates will be released as progress is made.

## What is in this repo?

This repo provides a way to perform inference with the Phyla-α model for your application. After performing the steps you will be able to give Phyla a fasta file and quickly get a phylogenetic tree. We are working on providing training code as well.


| Shorthand | Name in code           | Dataset | Description  |
|-----------|-----------------------------|---------|--------------|
| Phyla-α    | `phyla-alpha`       | 13,696 trees from OpenProteinSet  | Alpha release of Phyla meant as a proof of concept of ongoing work. |
| Phyla-β   | `phyla-beta`         | Full OpenProteinSet | Verion 2 of Phyla set to release in April after some methodological improvements and longer training. |


## Getting started with Phyla

### Step one: Install the enviornment

First you need to create an enviornment for mamba, following the instructions from their [Github](https://github.com/state-spaces/mamba) including the causal-conv1d package. I found installing this on a gpu helps get around some problems when installing. Once you can run this import without errors:

```python

from mamba_ssm import Mamba

```

then build the rest of the enviornment from [yaml file](https://github.com/mims-harvard/Phyla/blob/main/phyla/env/enviornment.yaml) provided in the envs folder in the phyla folder.

### Step two: Pip install the phyla package

Run 

```sh
 pip install -e .
```

from within this directory to install the Phyla package to your enviornment.

### Step three: Run the Phyla test.

Run "run_phyla_test.py" and if you get a tree printed out then everything is set up correctly! 

Once that is done just replace the fasta file in the run_phyla_test script to the fasta file with the protein sequences that you want to align and it will generate a tree.

## Some considerations

I have run this script on a H100 Nvidia-GPU, I believe it should work on a 32 GB V100 as well. The more memory you have in the GPU the larger the number of sequences you can generate trees for. 
To reconstruct the tree of life with 3,084 sequences, I had to run Phyla on cpus with ~1TB of memory. If you have an interest in running Phyla on a cpu to be able to run more sequences, raise an issue and we can add that functionality.

   
