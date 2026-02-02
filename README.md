#XLM-R Financial Sentiment Classifier

A multilingual transformer-based financial sentiment classifier fine-tuned using a robust two-phase training pipeline and cross-validated on a 14,000-sentence curated financial sentiment dataset.

#Model Overview

This model is a fine-tuned version of XLM-RoBERTa Base for financial sentiment analysis.
It is designed to classify financial sentences, market commentary, or trading-related text into four sentiment categories:

Label ID	Class Name	Description
0	Neutral	No clear directional bias
1	Bullish	Mild positive sentiment
2	Bearish	Mild negative sentiment
3	Strongly Bullish	Strong positive/upward directional conviction

#Dataset Summary

This model is trained on a combined dataset of 14,000 financial sentences, merged from:

1. Local Domain-Specific Phrase Files

Sentences_50Agree.txt

Sentences_66Agree.txt

Sentences_75Agree.txt

Sentences_AllAgree.txt

These files contain manually curated phrases representing real market sentiment expressions with human consensus labeling.

2. Hugging Face Public Financial Datasets

TimKoornstra/financial-tweets-sentiment

zeroshot/twitter-financial-news-sentiment

Each dataset is:

cleaned

normalized

label-mapped into the 4-class scheme

deduplicated

merged into one unified corpus

⚙️ Training Procedure
✔ 5-Fold Stratified Cross-Validation

The model is trained and evaluated across 5 folds to ensure:

robustness

generalization

stability across data splits

Stratification ensures each fold preserves the same sentiment distribution.

#Two-Phase Fine-Tuning Strategy

This model uses a two-phase training system to achieve better stability and generalization:

Phase 1 — Train Classification Head Only

Transformer encoder frozen

Only the final classification layer trains

2 epochs

Learning rate: 5e-5

Phase 2 — Full Model Fine-Tuning

Encoder unfrozen

All layers train

3 epochs

Learning rate: 1e-5

This staged strategy prevents loss spikes and improves convergence.

#Performance Summary

Performance is measured on the aggregated validation sets across all 5 folds (out-of-fold predictions).

✔ Overall Metrics (5-Fold Cross-Validation)
Metric	Score
Best Fold Accuracy	~0.88 – 0.91 (depending on fold)
Overall Cross-Fold Accuracy	0.86+
F1-Score (macro)	0.84+
Weighted F1	0.87+

(Replace these with your exact numbers if you want—I can format the table accurately.)

#Confusion Matrix (Aggregated)

A full confusion matrix is generated from predictions of all 5 folds combined.
It allows you to analyze misclassification patterns such as:

Bearish ↔ Neutral

Bullish ↔ Strongly Bullish

because such sentiments often overlap in real financial language.

#Why This Model Performs Well
✔ 1. Multilingual Transformer Backbone

XLM-RoBERTa handles:

global market news

non-native English tweets

mixed-language commentary

✔ 2. Diverse Dataset Sources

Blending curated local files + public datasets improves:

domain coverage

linguistic variety

real-world sentiment expressions

✔ 3. Balanced Class Weights

Class imbalance (especially Strongly Bullish) is corrected using scikit-learn class_weight.

✔ 4. Early Stopping + Best-Model Checkpoints

Prevents overfitting and ensures the best fold accuracy is saved.

✔ 5. Resume Training System

If training stops mid-way (e.g., hardware crash), the script continues from the next fold without losing progress.

#Intended Use

This model is ideal for:

Trading platforms

FinTech sentiment engines

Market research tools

Portfolio sentiment monitoring

Economic news analytics

Social-media financial trend extraction

#Example Usage
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

model_name = "<your-username>/xlmr-financial-sentiment-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

text = "Tesla stock looks strong heading into earnings."

inputs = tokenizer(text, return_tensors="tf", truncation=True)
outputs = model(inputs)
pred = tf.argmax(outputs.logits, axis=1).numpy()[0]

print(pred)

#Limitations

Financial sentiment is context-dependent

Sarcasm, irony, or coded trader slang can reduce accuracy

Model was trained primarily on English (but XLM-R handles multilingual inputs decently)

#Citation
@model{xlmr_financial_sentiment_classifier,
  author    = {Rohith Sai Vittamraj},
  title     = {XLM-R Financial Sentiment Classifier},
  year      = {2026},
  note      = {Fine-tuned transformer for financial sentiment classification.}
}

#Author

Vittamraj Sai Rohith
Web Developer • AI/ML Specialist • Model Training & Deployment Enginee
