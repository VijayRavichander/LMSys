
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
from config import notebook_config, model_config
import torch
import pandas as pd
from utils import process_text, test_tokenize, inference
from concurrent.futures import ThreadPoolExecutor


test = pd.read_csv(notebook_config.test_data_path)

cols = ["prompt", "response_a", "response_b"]

for item in cols:
    test.loc[:, item] = test[item].apply(process_text)

# print("Test DataFrame Head")
# display(test.head(3))

if notebook_config.train_notebook:
    model_saved = f"/kaggle/working/Merged-LoRA-v{notebook_config.version}-4bit"
else:
    model_saved = f"/kaggle/input/{}/Merged-LoRA-v{notebook_config.version}-4bit"


tokenizer = GemmaTokenizerFast.from_pretrained(model_saved)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"


data = pd.DataFrame()
data["id"] = test["id"]
data["input_ids"], data["attention_mask"] = test_tokenize(tokenizer, test["prompt"], test["response_a"], test["response_b"], model_config.max_length)
data["length"] = data["input_ids"].apply(len)

# DF with Res_A and Res_B switched to add variation to the input
aug_data = pd.DataFrame()
aug_data["id"] = test["id"]
aug_data["input_ids"], aug_data["attention_mask"] = test_tokenize(tokenizer, test["prompt"], test["response_b"], test["response_a"], model_config.max_length)
aug_data["length"] = aug_data["input_ids"].apply(len)


device_0 = torch.device('cuda:0')
model_0 = Gemma2ForSequenceClassification.from_pretrained(
    f"{model_saved}",
    device_map=device_0,
    use_cache=False,
)

device_1 = torch.device('cuda:1')
model_1 = Gemma2ForSequenceClassification.from_pretrained(
    f"{model_saved}",
    device_map=device_1,
    use_cache=False,
)

even_samples = data.iloc[0::2].copy()
odd_samples = data.iloc[1::2].copy()

with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(inference, (odd_samples, even_samples), (model_0, model_1), (tokenizer, tokenizer), (device_0, device_1), (2, 2), (3072, 3072))

result_df = pd.concat(list(results), axis = 0)


prob = result_df[['winner_model_a', 'winner_model_b', 'winner_tie']].values

result_df.loc[:, "winner_model_a"] = prob[:, 0]
result_df.loc[:, "winner_model_b"] = prob[:, 1]
result_df.loc[:, "winner_tie"] = prob[:, 2]
submission_df = result_df[["id", 'winner_model_a', 'winner_model_b', 'winner_tie']]
submission_df.to_csv('submission.csv', index=False)