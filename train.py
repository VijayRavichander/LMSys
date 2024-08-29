import torch
from config import notebook_config, model_config
from utils import CustomTokenizer, preprocess_data, compute_metrics
from model import CustomGemma2ForSequenceClassification
from transformers import (
    BitsAndBytesConfig,
    GemmaTokenizerFast,
    Gemma2Config,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
3

#----------------------------#
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
access_token = user_secrets.get_secret("hfkey")
login(token=access_token)


#-------------------------#
# DATA HANDLING

#Importing Gemma Tokenizer
tokenizer = GemmaTokenizerFast.from_pretrained(model_config.checkpoint)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

# Custom Tokenizer
encoder = CustomTokenizer(tokenizer, model_config.max_length)

# Creating HF Dataset
ds = preprocess_data()
ds = ds.map(encoder, batched = True, num_proc = 8)


# K Fold Split
if notebook_config.train_100_percent:
    folds = [
        (
            [i for i in range(len(ds))], 
            [i for i in range(len(ds)) if (i % model_config.n_splits == fold_idx)&(i<model_config.data_length)]
        ) 
        for fold_idx in range(model_config.n_splits)
    ]
    print("We are training with 100% data")
    
else:
    folds = [
        (
            [i for i in range(len(ds)) if i % model_config.n_splits != fold_idx],
            [i for i in range(len(ds)) if (i % model_config.n_splits == fold_idx)&(i<model_config.data_length)]
        ) 
        for fold_idx in range(model_config.n_splits)
        ]    

#---------------------------------------------------------------------------#
#TRAING ARGS
    
training_args = TrainingArguments(
    output_dir = model_config.output_dir, 
    overwrite_output_dir = True, 
    report_to = "none", 
    num_train_epochs = model_config.n_epochs, 
    per_device_train_batch_size = model_config.per_device_train_batch_size, 
    gradient_accumulation_steps = model_config.gradient_accumulation_steps, 
    per_device_eval_batch_size = model_config.per_device_eval_batch_size, 
#     gradient_checkpointing = True, 
    logging_steps = 10, 
    eval_strategy = 'epoch', 
    save_strategy = 'no', 
    optim = model_config.optim_type, 
    fp16 = True, 
    learning_rate =  model_config.lr, 
    warmup_steps = model_config.warmup_steps,
    metric_for_best_model = 'log_loss', 
    greater_is_better = False
)

#-----------------------------------------------------------
# LORA CONFIG

lora_config = LoraConfig(
    r = model_config.lora_r, 
    lora_alpha = model_config.lora_alpha, 
    target_modules = ["q_proj", "k_proj", "v_proj", "down_proj", "up_proj", "o_proj", "gate_proj"], 
    layers_to_transform = [i for i in range(42) if i >= model_config.freeze_layers], 
    lora_dropout = model_config.lora_dropout, 
    bias = model_config.lora_bias, 
    task_type = TaskType.SEQ_CLS, 
    modules_to_save = ["score", "classifier_head1", "classifier_head2"]
)

#---------------------------------------------------------------------------#
#QLORA CONFIG

qlora = {}
if notebook_config.use_qlora:
    bnb_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    bnb_4bit_quant_type = 'nf4', 
    bnb_4bit_use_double_quant = False, 
    bnb_4bit_compute_dtype = torch.float16, 
    llm_int8_skip_modules = ["score", "classifier_head1", "classifier_head2"]
    )
    qlora['quantization_config'] = bnb_config
    print("Using QLoRA")

#--------------------#

config2 = Gemma2Config.from_pretrained(model_config.checkpoint)
config2.num_labels = 3
model = CustomGemma2ForSequenceClassification.from_pretrained(
    model_config.checkpoint,
    config=config2,
    num_labels_head1=58,
    num_labels_head2=58,
    torch_dtype=torch.float16,
    device_map="auto",
    **qlora
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print("Model Params Config")
print(model)
print(model.print_trainable_parameters())


train_idx, eval_idx = folds[model_config.fold_idx]

trainer = Trainer(
    args=training_args, 
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds.select(train_idx[train_idx]),
    eval_dataset=ds.select(eval_idx[eval_idx]),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train()
trainer.save_model(f"./output/LoRA-v{notebook_config.version}")