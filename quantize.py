
# Quantize & Merge the trained and saved model
import torch
import os
from peft import PeftModel
from config import model_config, notebook_config
from transformers import GemmaTokenizerFast,Gemma2Config,BitsAndBytesConfig
from model import CustomGemma2ForSequenceClassification
from peft import PeftModel

#Load the tokenizer
tokenizer = GemmaTokenizerFast.from_pretrained(model_config.checkpoint)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

# Load the Trained LoRA Model
config2 = Gemma2Config.from_pretrained(model_config.checkpoint)
config2.num_labels = 3
model = CustomGemma2ForSequenceClassification.from_pretrained(
    model_config.checkpoint,
    config=config2,
    num_labels_head1=58,
    num_labels_head2=58,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Loading the Adapters
model.config.use_cache = False
model = PeftModel.from_pretrained(model, f"/kaggle/working/output/LoRA-v{notebook_config.version}")

# Merging LoRa Weights to the Base fp16
model = model.merge_and_unload()

# Save the Merged Model
model.save_pretrained(f"Merged-LoRA-v{notebook_config.version}") 
tokenizer.save_pretrained(f"Merged-LoRA-v{notebook_config.version}")

# Clearing the RAM
del model
torch.cuda.empty_cache()

# Quantizing the whole model to nf4 and saving
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4", #nf4 or fp4
    bnb_4bit_use_double_quant = False,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules = ["score","classifier_head1", "classifier_head2"]
)

config2 = Gemma2Config.from_pretrained(model_config.checkpoint)
config2.num_labels = 3
model = CustomGemma2ForSequenceClassification.from_pretrained(
    f"Merged-LoRA-v{notebook_config.version}",
    config=config2,
    num_labels_head1=58,
    num_labels_head2=58,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config = bnb_config,
)
model.config.use_cache = False

# Remove the LoRA Non Quantized Saved Model
os.system(f"rm -r Merged-LoRA-v{notebook_config.version}") 

# Saving the Model 
model.save_pretrained(f"Merged-LoRA-v{notebook_config.version}-4bit") 
tokenizer.save_pretrained(f"Merged-LoRA-v{notebook_config.version}-4bit")