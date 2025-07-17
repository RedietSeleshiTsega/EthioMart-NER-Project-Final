EthioMart Amharic NER Project
🚀 Tasks Covered
✅ Task 1: Amharic Data Collection and Preprocessing
✅ Task 2: Labeling Amharic Text in CoNLL Format
🧠 Business Context
EthioMart aims to be the primary hub for Telegram-based e-commerce in Ethiopia. Currently, products, prices, and contact info are scattered across various Telegram channels. This project solves that problem by building an Amharic NER pipeline that extracts:

🛍️ Product Names
💵 Prices
📍 Locations
...from Telegram posts to power EthioMart’s unified product catalog.

🧪 Task 1: Data Ingestion and Preprocessing
1.1 Telegram Channel Selection
Selected Channels:

@shageronlinestore
@leyueqa
@fashiontera
@helloomarketethiopia
@kuruwear
1.2 Ingestion Pipeline Overview
Built using telethon and Telegram API.
Authenticated via https://my.telegram.org
Collected messages, timestamps, metadata, images, and docs.
1.3 Preprocessing Pipeline
Steps:

✅ Normalize Amharic text (remove emojis, symbols, etc.)
✅ Tokenize sentences
✅ Clean empty/image-only rows
✅ Structured into CSV for NER modeling
Output File: data/processed/processed_data.csv

🏷️ Task 2: NER Labeling in CoNLL Format
2.1 Objective
Label 50 Amharic Telegram messages to create a custom NER dataset.

2.2 Entity Types
Entity	Labels Used
Product	B-Product, I-Product
Price	B-PRICE, I-PRICE
Location	B-LOC, I-LOC
Others	O
2.3 Labeling Process
Tokens exported from processed_data.csv
Manual annotation using Excel file: labeling_template.xlsx
Converted to CoNLL format: conll_labeled.txt
Sample Format:

spa     B-Product  
gel     I-Product  
socks   I-Product  
ዋጋ     B-PRICE  
400     I-PRICE  
ብር     I-PRICE
🧾 Output Files
File	Description
processed_data.csv	Cleaned and tokenized text data
labeling_template.xlsx	Manually labeled NER Excel sheet
conll_labeled.txt	CoNLL format NER dataset
📌 Next Steps
Fine-tune xlm-roberta-base on CoNLL dataset
Evaluate using F1, Precision, Recall
Apply SHAP/LIME for interpretability
Compare and recommend best model
👥 Contributors
Rediet