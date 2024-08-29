import os
from dataclasses import dataclass
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

@dataclass
class NotebookConfig:
    version : int = 1
    use_qlora : bool = True
    train_100_percent : bool = False
    add_extra_data : bool = False
    debug : bool = True
    main_data_path : str = f"./data/train.csv"
    suppl_data_path : str = ""
    

@dataclass
class ModelConfig: 
    # File 
    output_dir: str = f"output"
    checkpoint: str = "google/gemma-2-9b"
    
    # Data Loading
    data_length: int = 0
    max_length: int = 2048
    n_splits: int = 5
    fold_idx: int = 0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = 4
    
    # Optimization
    optim_type: str = "adamw_8bit"
    n_epochs: int = 1
    freeze_layers: int = 0
    lr: float = 2e-4
    warmup_steps: int = 20
        
    # LoRa 
    lora_r: int = 64
    lora_alpha: float = 4
    lora_dropout: float = 0.05
    lora_bias: str = "none"

notebook_config = NotebookConfig()
model_config = ModelConfig()