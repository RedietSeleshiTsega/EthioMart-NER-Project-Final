from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
from utils import get_label_list

model_path = "models/bert-tiny-amharic"  
label_list = get_label_list()


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()


text = "ዶክተር ሃና በአዲስ አበባ ታህሳስ 20 ወደ ብሪጃ ሆስፒታል ተማረከች።"


inputs = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
outputs = model(**inputs).logits
predictions = torch.argmax(outputs, dim=2)


tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
pred_labels = [label_list[p.item()] for p in predictions[0]]


print("\nNamed Entity Recognition:")
for token, label in zip(tokens, pred_labels):
    if label != "O":
        print(f"{token}: {label}")
