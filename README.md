# Phyla: Towards a Foundation Model for Phylogenetic Inference

![Tree of life](img/16S_sequences.png)

## What is Phyla? 

Phyla is a protein language model designed to model both individual sequences and inter-sequence relationships. It leverages a hybrid state-space transformer architecture and is trained on two tasks: masked language modeling and phylogenetic tree reconstruction using sequence embeddings. Phyla enables rapid construction of phylogenetic trees of protein sequences, offering insights that differ from classical methods in potentially functionally significant ways.

## Disclaimer

We are excited to introduce Phyla-β, an early-stage version of our model that is still under active development. Future iterations will incorporate methodological improvements and additional training data as we continue refining the model. Please note that this work is ongoing, and updates will be released as progress is made.

## What is in this repo?

This repo provides a way to perform inference with the Phyla-α/Phyla-β model for your application. After performing the steps you will be able to give Phyla a fasta file and quickly get a phylogenetic tree. We are working on providing training code as well.


| Shorthand | Name in code           | Dataset | Description  |
|-----------|-----------------------------|---------|--------------|
| Phyla-α    | `phyla-alpha`       | 13,696 trees from OpenProteinSet  | Alpha release of Phyla meant as a proof of concept of ongoing work. |
| Phyla-β   | `phyla-beta`         | 3,321 high-quality trees from OpenProteinSet | Beta release of Phyla improving on Phyla-α by training on better data and improved tree loss. |

## What is the difference between Phyla-α and Phyla-β?

After releasing Phyla-α we revised our tree loss and retrained our model on a cleaned version of OpenProteinSet, using the methodologies for MSA cleaning introduced by [EVE](https://github.com/OATML-Markslab/EVE). We also found surprisingly that masked langugae modeling decreased performance, so this was removed in training. In benchmarking (see Evolutionary Reasoning Benchmark section) Phyla-β is better than Phyla-α and should be used for all applications. It is also more lightweight than Phyla-α which allows for longer inputs.

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

## System Requirements and Scalability 

This script has been tested on an H100 Nvidia GPU and is expected to work on a 32 GB V100 as well. Greater GPU memory capacity allows for generating trees for a larger number of sequences. Reconstructing the tree of life with 3,084 sequences required running Phyla on CPUs with approximately 1 TB of memory. For those interested in running Phyla on a CPU to handle more sequences, raising an issue will help prioritize the addition of that functionality.

# Tree Reasoning Benchmark

The **Tree Reasoning Benchmark** consists of two tasks across three datasets. It evaluates a model's ability in:

1. **Phylogenetic Tree Reconstruction**  
   - Measured by **normalized Robinson-Foulds distance** (**norm-RF**).

2. **Taxonomic Clustering**  
   - Measured by **cluster completeness** and **Normalized Mutual Information** (**NMI**).

---

## Task 1: Tree Reconstruction

**Benchmarking Approach:**

- Compute the **pairwise distance matrix** from protein embeddings.
- Use the **Neighbor Joining algorithm** to construct a phylogenetic tree.
- Compare against the ground truth using **norm-RF**.

**Datasets Used:**

- `TreeFam` found in a pickle file here: https://dataverse.harvard.edu/api/access/datafile/11564365
- `TreeBASE` found in a zip file here: https://dataverse.harvard.edu/api/access/datafile/11564367

### TreeBASE Details

- After downloading and unzipping, you'll find two directories: `sequences/` and `trees/`.
- Filenames are aligned: for example, the tree for `TB2/S137_processed.fa` in `sequences/` is `TB2/S137_processed_tree.nh` in `trees/`.

### TreeFam Details

- After unzipping, you receive a **pickle file**.
- Each key corresponds to a sequence/tree name.
- Each entry contains:
  - `sequences`: the protein sequences
  - `tree_newick`: the Newick-formatted tree

> ⚠️ **Note**: Trees from TreeFam require formatting before use. A formatting script is provided in the evaluation code. If there is enough interest in the benchmark, preprocessed trees can be generated to avoid this step.

---

## Task 2: Taxonomic Clustering

**Benchmarking Approach:**

- Perform **k-means clustering** on protein embeddings.
- Evaluate using:
  - **Cluster Completeness**
  - **Normalized Mutual Information (NMI)**

**Dataset Used:**

- `GTDB` (Genome Taxonomy Database) found in a tar-file here: https://dataverse.harvard.edu/api/access/datafile/11564368

### GTDB Details

- After extracting the tar file, you'll find a series of `.tsv` and `.pickle` files with names like:
  - `sampled_bac120_taxonomy_class_0.tsv`
  - `sampled_bac120_taxonomy_class_sequences_0.pickle`

- The structure of these filenames follows the format:
  - `sampled_bac120_taxonomy_[level]_[replicate].tsv`
  - `sampled_bac120_taxonomy_[level]_sequences_[replicate].pickle`

  Where:
  - `[level]` refers to the **taxonomic rank** (e.g., class, order, family).
  - `[replicate]` is a numeric index indicating a random sampling replicate.

- Each `.tsv` file contains:
  - Sequence names
  - Taxonomic labels at the specified level

- The corresponding `.pickle` file contains the actual sequences for those entries.

> Each replicate includes random groupings of 50 distinct labels, with 10 sequences per label. Use the taxonomic column in the `.tsv` and the sequence names to extract the clustering labels.

## Task 3: Functional Prediction

This task evaluates how well a model can predict functional impacts of protein variants using data from the **ProteinGym DMS Substitution Benchmark**. We use 83 datasets selected to fit within the memory limits of a single H100 GPU, with performance measured by **Spearman correlation**.

**Benchmarking Approach:**

- For all baseline protein language models, a **linear probe** is trained on the model embeddings to predict variant effects.
- For **Phyla**, the process involves:
  1. Constructing a phylogenetic tree from the protein sequences.
  2. Injecting known functional labels into the corresponding tree leaves.
  3. Using **TreeCluster** to cluster the tree into clades.
  4. Assigning predicted labels to unlabeled leaves by averaging the known labels in their clade.

This tree-based propagation strategy yields the best Spearman correlation for Phyla in functional prediction.

> We use the [TreeCluster](https://github.com/niemasd/TreeCluster) toolkit to perform tree clustering.

---

# Evaluation Instructions

To evaluate tree reconstruction, taxonomic clustering, or functional prediction, run:

```bash
python -m eval.evo_reasoning_eval configs/sample_eval_config.yaml
```

### Modifying the Config

Open and modify `configs/sample_eval_config.yaml` as needed:

#### 1. `trainer.model_type`

Choose the model to run:

- `Phyla-beta` (default Phyla model)
- `ESM2` (ESM2 650M)
- `ESM2_3B` (ESM2 3B)
- `ESM3`
- `EVO`
- `PROGEN2_LARGE`
- `PROGEN2_XLARGE`

#### 2. `dataset.dataset`

Set one of the following datasets:

- `treebase` – For tree reconstruction (TreeBASE)
- `treefam` – For tree reconstruction (TreeFam)
- `GTB` – For taxonomic clustering (GTDB)
- `protein_gym` – For functional prediction

> Required files will be downloaded automatically.

#### 3. `evaluating.device`

Set the GPU device to use (e.g., `"cuda:0"`, `"cuda:5"`).

#### 4. `evaluating.random`

Set this to `true` to evaluate a randomly initialized model (default is `false`).

   
