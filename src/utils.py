
import re
import torch
import pandas as pd
from spacy.lang.en import English
from transformers import AutoTokenizer, DataCollatorForTokenClassification

# Natural Language Processing Setup
nlp = English()
INFERENCE_MAX_LENGTH = 3500
threshold = 0.99

# Regular expressions for detecting various types of PII.
email_regex = re.compile(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+')
phone_num_regex = re.compile(r"(\\(\\d{3}\\)\\d{3}\\-\\d{4}\\w*|\\d{3}\\.\\d{3}\\.\\d{4})\\s")
url_regex = re.compile(
    r'http[s]?://'
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\\.)+(?:[A-Z]{2,6}\\.?|[A-Z0-9-]{2,}\\.?)|'
    r'localhost|'
    r'\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})'
    r'(?::\\d+)?'
    r'(?:/?|[/?]\\S+)', re.IGNORECASE)
street_regex = re.compile(r'\\d{1,4} [\\w\\s]{1,20}(?:street|apt|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\\W?(?=\\s|$)', re.IGNORECASE)

def find_span(target: list[str], document: list[str]) -> list[list[int]]:
    idx = 0
    spans = []
    span = []
    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue
    return spans

def process_predictions(flattened_preds):
    predictions_softmax_all = []
    for predictions in flattened_preds:
        predictions_softmax = torch.softmax(predictions, dim=-1)
        predictions_softmax_all.append(predictions_softmax)
    return predictions_softmax_all

def process_predictions_ans(flattened_preds, threshold=0.95):
    preds_final = []
    for predictions in flattened_preds:
        predictions_softmax = predictions
        predictions_argmax = predictions.argmax(-1)
        predictions_without_O = predictions_softmax[:, :12].argmax(-1)
        O_predictions = predictions_softmax[:, 12]
        pred_final = torch.where(O_predictions < threshold, predictions_without_O, predictions_argmax)
        preds_final.append(pred_final.numpy())
    return preds_final

def tokenize_row(example, tokenizer, config):
    text = []
    token_map = []
    idx = 0
    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):
        text.append(t)
        token_map.extend([idx]*len(t))
        if ws:
            text.append(" ")
            token_map.append(-1)
        idx += 1
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation=config.truncation, max_length=config.max_length)
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "offset_mapping": tokenized.offset_mapping,
        "token_map": token_map,
    }

def downsample_df(train_df, percent):
    train_df['is_labels'] = train_df['labels'].apply(lambda labels: any(label != 'O' for label in labels))
    true_samples = train_df[train_df['is_labels'] == True]
    false_samples = train_df[train_df['is_labels'] == False]
    n_false_samples = int(len(false_samples) * percent)
    downsampled_false_samples = false_samples.sample(n=n_false_samples, random_state=42)
    downsampled_df = pd.concat([true_samples, downsampled_false_samples])
    return downsampled_df


