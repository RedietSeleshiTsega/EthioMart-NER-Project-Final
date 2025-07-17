import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from datasets import load_dataset, load_metric
from utils import get_label_list, tokenize_and_align_labels

model_path = "models/bert-tiny-amharic" 
label_list = get_label_list()
num_labels = len(label_list)


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels)


dataset = load_dataset("csv", data_files={"test": "./data/ner_dataset_test.csv"}, delimiter=",")
tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label_list), batched=True)


metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results


trainer = Trainer(model=model)
results = trainer.evaluate(tokenized_dataset["test"])
print("Evaluation Results:", results)
