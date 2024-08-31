# Handling Data 
import pandas as pd
import json
import numpy as np
import torch
from config import notebook_config, model_config
from datasets import Dataset
from transformers import EvalPrediction
from sklearn.metrics import log_loss, accuracy_score
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

def preprocess_data():
    df = pd.read_csv(notebook_config.main_data_path)

    df["id"] = df["id"].astype('str')

    # Add the length to Model_Config Object
    model_config.data_length = len(df)

    # Renaming Labels
    df.loc[df.model_a == "claude-1", 'model_a'] = 'claude-v1'
    df.loc[df.model_b == "claude-1", 'model_b'] = 'claude-v1'

    df.loc[df.model_a == "claude-instant-1", 'model_a'] = 'claude-instant-v1'
    df.loc[df.model_b == "claude-instant-1", 'model_b'] = 'claude-instant-v1'

    df.loc[df.model_a.str.contains("gpt-3.5"), "model_a"] = "gpt-3.5-turbo"
    df.loc[df.model_b.str.contains("gpt-3.5"), "model_b"] = "gpt-3.5-turbo"

    df.loc[df.model_a.str.contains("gpt-4"), "model_a"] = "gpt-4"
    df.loc[df.model_b.str.contains("gpt-4"), "model_b"] = "gpt-4"


    if notebook_config.add_extra_data:
        df2 = pd.read_csv(notebook_config.suppl_data_path)
        df = pd.concat([df, df2], axis = 0, ignore_index = True)

    # Using few points for debugging the code
    if notebook_config.debug:
        sample_points = 64
        df = df.iloc[:sample_points]
        
    # Label Encoding
    label1 = df.model_a.unique()
    label2 = df.model_b.unique()

    label = np.union1d(label1, label2)
    label = sorted(label)

    print(f"Unique Labels: {len(label)}")

    label_map = {y:x  for x, y in enumerate(label)}

    df.model_a = df.model_a.map(label_map).astype("int32")
    df.model_b = df.model_b.map(label_map).astype("int32")

    # Create a HF Dataset
    print("Creating HF Dataset")
    ds = Dataset.from_pandas(df)

    return ds

class CustomTokenizer: 
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def prepare_text(self, prompts, responses_a, responses_b):
        
        prompts = json.loads(prompts)
        responses_a = json.loads(responses_a)
        responses_b = json.loads(responses_b)
        
         #print(len(prompts)) This is kind of random. Anywhere between 1 - num_procs(8 here)
        
        rounds = [
            f"<start_of_turn>prompt\n{prompts[i]}<end_of_turn>\n"
            +f"<start_of_turn>response_a\n{responses_a[i]}<end_of_turn>\n"
            +f"<start_of_turn>response_b\n{responses_b[i]}<end_of_turn>"
            for i in range(len(prompts))
        ]
        
        tmp = "\n".join(rounds)
        for k in range(len(rounds)):
            tmp = "\n".join(rounds[k:])
            if len(self.tokenizer(tmp)["input_ids"]) < self.max_length:
                break
                
        return tmp
    
    def __call__(self, batch: dict) -> dict:  
        
#        print(len(batch["prompt"])) # 8
        
        texts = [
            self.prepare_text(p, r_a, r_b)
            for p, r_a, r_b in zip(batch["prompt"], batch["response_a"], batch["response_b"])
        ]
        
        tokenized = self.tokenizer(texts, max_length = self.max_length, truncation = True)
        labels = []
        for a_win, b_win, draw, c, d in zip(batch["winner_model_a"], batch["winner_model_b"], batch["winner_tie"], batch["model_a"], batch["model_b"]):
            if a_win:
                label = 0
                
            elif b_win:
                label = 1
                
            else: 
                label = 2
            
            labels.append((label, c, d))
        
        return {**tokenized, "labels": labels}
    
# Computing Loss 
def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    
    #Split the Preds
    preds_head1 = preds[0] # Winner Pred
    preds_head2 = preds[1] # Model A Pred
    preds_head3 = preds[2] # Model B Pred
    
    labels_head1 = labels[: , 0] # Winner Label
    labels_head2 = labels[: , 1] # Model A Label
    labels_head3 = labels[: , 2] # Model B Label 
    
      # Compute log loss and accuracy for each head
    probs_head1 = torch.from_numpy(preds_head1).float().softmax(-1).numpy()
    loss_head1 = log_loss(y_true=labels_head1, y_pred=probs_head1, labels=[x for x in range(3)])
    acc_head1 = accuracy_score(y_true=labels_head1, y_pred=preds_head1.argmax(-1))
    
    probs_head2 = torch.from_numpy(preds_head2).float().softmax(-1).numpy()
    loss_head2 = log_loss(y_true=labels_head2, y_pred=probs_head2, labels=[x for x in range(58)])
    acc_head2 = accuracy_score(y_true=labels_head2, y_pred=preds_head2.argmax(-1))

    probs_head3 = torch.from_numpy(preds_head3).float().softmax(-1).numpy()
    loss_head3 = log_loss(y_true=labels_head3, y_pred=probs_head3, labels=[x for x in range(58)])
    acc_head3 = accuracy_score(y_true=labels_head3, y_pred=preds_head3.argmax(-1))
    
    # Return the metrics for each head
    return {
        "acc_classify": acc_head1,
        "log_loss_classify": loss_head1,
        "acc_model_a": acc_head2,
        "log_loss_model_a": loss_head2,
        "acc_model_b": acc_head3,
        "log_loss_model_b": loss_head3
    }


def process_text(text):
    return json.loads(text)


def test_tokenize(tokenizer, prompt, res_a, res_b, max_length):
    text = []
    
    for p, a, b in zip(prompt, res_a, res_b):
        
        rounds = [
            f"<start_of_turn>prompt\n{p[i]}<end_of_turn>\n" + 
            f"<start_of_turn>prompt\n{a[i]}<end_of_turn>\n" + 
            f"<start_of_turn>prompt\n{b[i]}<end_of_turn>"
            for i in range(len(p))
        ]
    
        tmp = "\n".join(rounds)
        for k in range(len(rounds)):
            tmp = "\n".join(rounds[k:])
            if len(tokenizer(tmp)['input_ids']) < max_length:
                break
        
        text.append(tmp)
    
    tokenized = tokenizer(text, max_length = max_length, truncation = True, padding = False)
    inputs_ids = tokenized.input_ids
    attn_mask = tokenized.attention_mask
    
    return inputs_ids, attn_mask


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, tokenizer, device, batch_size, max_length):
    a_win, b_win, tie_win = [], [], []
    
    print(f"DF {df}")
    for start_idx in range(0, len(df), batch_size):
        
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx : end_idx]
        
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        
        inputs = pad_without_fast_tokenizer_warning(
                tokenizer, 
                {'input_ids': input_ids, 'attention_mask': attention_mask}, 
                padding = 'longest', 
                pad_to_multiple_of = None,
                return_tensors = 'pt'
                )
        
        output = model(**inputs.to(device))
        prob = output.logits.softmax(-1).cpu()
                
        a_win.extend(prob[:, 0].tolist())
        b_win.extend(prob[:, 1].tolist())
        tie_win.extend(prob[:, 2].tolist())
        
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie_win
    
    return df