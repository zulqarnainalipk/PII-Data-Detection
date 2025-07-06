import gc
import json
import torch
import pandas as pd
import onnxruntime
from itertools import chain
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification

from config import Config
from utils import find_span, process_predictions, process_predictions_ans, tokenize_row, downsample_df, email_regex, phone_num_regex, url_regex, nlp

# Global flags
debug_on_train_df = False
convert_before_inference = False

# Load data
data = json.load(open(Config.train_dataset_path))
test_data = json.load(open(Config.test_dataset_path))

# Prepare labels
all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {v: k for k, v in label2id.items()}

# Initialize tokenizer
first_model_path = list(Config.model_paths.keys())[0]
tokenizer = AutoTokenizer.from_pretrained(first_model_path)

# Create DataFrames
df_train = pd.DataFrame(data)
df_train["fold"] = df_train["document"] % 4
df_test = pd.DataFrame(test_data)

# Tokenize and save datasets
if debug_on_train_df:
    if Config.load_from_disk is None:
        for i in range(-1, 4):
            train_df = df_train[df_train["fold"] == i].reset_index(drop=True)
            if i != Config.trn_fold and Config.downsample > 0:
                train_df = downsample_df(train_df, Config.downsample)
            ds = Dataset.from_pandas(train_df)
            ds = ds.map(
                lambda example: tokenize_row(example, tokenizer, Config),
                batched=False,
                num_proc=2,
                desc="Tokenizing",
            )
            ds.save_to_disk(f"{Config.save_dir}fold_{i}.dataset")
else:
    if Config.load_from_disk is None:
        ds = Dataset.from_pandas(df_test)
        ds = ds.map(
            lambda example: tokenize_row(example, tokenizer, Config),
            batched=False,
            num_proc=2,
            desc="Tokenizing",
        )
        ds.save_to_disk(f"{Config.save_dir}test.dataset")

# Prepare DataLoader
keep_cols = {"input_ids", "attention_mask"}
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=512)

if not debug_on_train_df:
    test_ds = load_from_disk(f'{Config.save_dir}test.dataset')
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])
    Config.data_length = len(test_ds)
    Config.len_token = len(tokenizer)
    test_dataloader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=False, collate_fn=collator)
else:
    fold = Config.trn_fold
    test_ds = load_from_disk(f'{Config.save_dir}fold_{fold}.dataset')
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])
    Config.data_length = len(test_ds)
    Config.len_token = len(tokenizer)
    test_dataloader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=False, collate_fn=collator)

# Prediction and Ensemble
predictions_softmax_logits = []
for model_path, weight in Config.model_paths.items():
    fold = Config.trn_fold
    if convert_before_inference:
        # This part requires onnxconverter_common and is skipped for now
        pass
    else:
        quantized_model_name = Config.converted_path + f"/optimized{model_path.split('/')[-1]}_f{fold}.onnx"

    session = onnxruntime.InferenceSession(quantized_model_name, providers=["CUDAExecutionProvider"])
    
    # The predict function from the notebook needs to be adapted to use the session
    # For simplicity, I'm assuming a direct call to a predict function that takes session and dataloader
    # This part needs the actual predict function from the notebook
    # For now, let's mock it or assume it's imported from utils if it was there
    
    # Mocking the predict function for now
    def predict(data_loader, session, config):
        # This is a placeholder. The actual logic from the notebook's predict function should go here.
        # It should iterate through data_loader, run session.run, and return processed predictions.
        # For demonstration, returning dummy data.
        dummy_predictions = []
        for batch in data_loader:
            # Simulate model output
            dummy_output = torch.randn(Config.batch_size, Config.max_length, len(all_labels))
            dummy_predictions.append(dummy_output)
        
        flattened_preds = [logit for batch in dummy_predictions for logit in batch]
        return process_predictions(flattened_preds)

    predictions_softmax_all = predict(test_dataloader, session, Config)
    predictions_softmax_logits.append(predictions_softmax_all)

del test_dataloader, test_ds
gc.collect()
torch.cuda.empty_cache()

# Calculate weighted mean of predictions
predictions_mean_all = []
total_weight = sum(Config.model_paths.values())
model_weights = list(Config.model_paths.values())

for sample_index in range(len(predictions_softmax_logits[0])):
    weighted_predictions_sum = torch.zeros(predictions_softmax_logits[0][sample_index].size())
    for model_index in range(len(predictions_softmax_logits)):
        weighted_prediction = predictions_softmax_logits[model_index][sample_index] * (model_weights[model_index] / total_weight)
        weighted_predictions_sum += weighted_prediction
    predictions_mean_all.append(weighted_predictions_sum)

# Process final predictions
processed_predictions = process_predictions_ans(predictions_mean_all, threshold=Config.threshold)

# Extract information and create submission file
triplets = []
pairs = set()
processed = []
emails = []
phone_nums = []
urls = []
streets = []

# Re-load ds to get original data for extraction
ds = load_from_disk(f'{Config.save_dir}test.dataset')

for p, token_map, offsets, tokens, doc, full_text in zip(
    processed_predictions,
    ds["token_map"],
    ds["offset_mapping"],
    ds["tokens"],
    ds["document"],
    ds["full_text"]
):
    for token_pred, (start_idx, end_idx) in zip(p, offsets):
        label_pred = id2label[token_pred]
        if start_idx + end_idx == 0 or label_pred == "O":
            continue
        if token_map[start_idx] == -1:
            start_idx += 1
        while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
            start_idx += 1
        if start_idx >= len(token_map):
            break
        token_id = token_map[start_idx]
        if label_pred in ("O", "B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM") or token_id == -1:
            continue
        pair = (doc, token_id)
        if pair not in pairs:
            processed.append({"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id]})
            pairs.add(pair)
    
    for token_idx, token in enumerate(tokens):
        if re.fullmatch(email_regex, token) is not None:
            emails.append(
                {"document": doc, "token": token_idx, "label": "B-EMAIL", "token_str": token}
            )
                
    matches = phone_num_regex.findall(full_text)
    if matches:
        for match in matches:
            target = [t.text for t in nlp.tokenizer(match)]
            matched_spans = find_span(target, tokens)
            for matched_span in matched_spans:
                for intermediate, token_idx in enumerate(matched_span):
                    prefix = "I" if intermediate else "B"
                    phone_nums.append(
                        {"document": doc, "token": token_idx, "label": f"{prefix}-PHONE_NUM", "token_str": tokens[token_idx]}
                    )
    
    matches = url_regex.findall(full_text)
    if matches:
        for match in matches:
            target = [t.text for t in nlp.tokenizer(match)]
            matched_spans = find_span(target, tokens)
            for matched_span in matched_spans:
                for intermediate, token_idx in enumerate(matched_span):
                    prefix = "I" if intermediate else "B"
                    urls.append(
                        {"document": doc, "token": token_idx, "label": f"{prefix}-URL_PERSONAL", "token_str": tokens[token_idx]}
                    )

df = pd.DataFrame(processed + phone_nums + emails + urls)
df["row_id"] = list(range(len(df)))
df[["row_id", "document", "token", "label"]].to_csv("submission.csv", index=False)


