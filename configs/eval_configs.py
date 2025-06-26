class TrainerConfig():
    lr: float = 0.0001
    record: bool = False
    epochs: int = 6000
    epochs_callback: int = 1000
    steps_callback: int = 10000
    #Number of sub trees per batch
    batch_size: int = 2
    #Number of sequences per sub tree
    sub_tree_size: int = 10
    save_path: str = 'defaultpath'
    scheduler: str = 'defaultpath' # Change to 'cosine_annealing' for scheduler
    model_type: str = 'MAMBA'
    checkpoint_path: str = 'defaultpath'
    checkpoint_id: int = 0  
    ddp: bool = False
    deepspeed: bool = False
    num_annealing_steps: int = 10000
    num_warmup_steps: int = 1000
    use_tree_loss: bool = True
    use_mlm_loss: bool = True
    resume: bool = False
    run_name: str = 'default_run'
    symmetry_loss: int = 0

class DatasetConfig():
    dataset: str = 'zf10'
    #Number of sub trees per batch
    batch_size: int = 2
    #Number of sequences per sub tree
    sub_tree_size: int = 10
    #Detect what gpu you are on, make tree/trees in range of 0 to largest sub-tree size for that gpu
    adaptive_batch_size: bool = False

class Mamba_ModelConfig():
    d_model: int = 256
    n_layer: int = 32 # Change based on number of layers in input
    vocab_size: int = 24
    ssm_cfg: dict = {}
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    num_blocks: int = 1 # Change based on number of layers in input
    model_name: str = 'MAMBA'
    calculation_method: str = "attention"
    use_attention: bool = True
    positional_embeddings: bool = False
    inject_rotary_attention: bool = False
    bidirectional_strategy: str = "add"
    bidirectional_weight_tie: bool = True
    bidirectional: bool = False
    ranking_loss: bool = False

class EvalConfig():
    random: bool = False #Flag to randomly initialize model
    convert_to_aa: bool = False
    extra_name: str = ""
    chunk: list = [] #A list where first entry is number of chunks and second is the specific chunk number to run
    device: str = 'cuda:5'

class ESM2_ModelConfig():
    model_name: str = "ESM2"

class EVO_ModelConfig():
    model_name: str = "EVO"

class ESM3_ModelConfig():
    model_name: str = "ESM3"

class ESM2_3B_ModelConfig():
    model_name: str = "ESM2_3B"

class Config():
    model: model = Mamba_ModelConfig()
    dataset: dataset = DatasetConfig()
    trainer: trainer = TrainerConfig()
    eval: eval = EvalConfig()
