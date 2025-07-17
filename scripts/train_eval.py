from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
import numpy as np
import os

# Setup model paths
models = {
    "bert": "Davlan/bert-tiny-amharic",
    "xlmr": "xlm-roberta-base"
}

# Load dataset
dataset = load_dataset("json", data_files={
    "train": "../data/train.json",
    "validation": "../data/val.json",
    "test": "../data/test.json"
})

label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

metric = load_metric("seqeval")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(preds_row, labels_row) if l != -100]
        for preds_row, labels_row in zip(preds, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(preds_row, labels_row) if l != -100]
        for preds_row, labels_row in zip(preds, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

for name, model_path in models.items():
    print(f"Training: {name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels)

    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        word_ids = tokenized_inputs.word_ids()
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != prev_word_idx:
                labels.append(example["ner_tags"][word_idx])
            else:
                labels.append(-100)
            prev_word_idx = word_idx
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    training_args = TrainingArguments(
        output_dir=f"../results/{name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"../results/{name}/logs",
        logging_steps=10,
        report_to="none"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate(tokenized_datasets["test"])
