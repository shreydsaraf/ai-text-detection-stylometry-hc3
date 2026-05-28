# AI Text Detection Using Stylometric Features on HC3

This repository contains the code and results for an AI-generated text detection experiment using the public HC3 Human ChatGPT Comparison Corpus.

The project compares TF-IDF baselines, stylometric-only models, and a hybrid TF-IDF + stylometric model. It also includes an ablation study to evaluate which stylometric feature groups contribute most to classification performance.

## Dataset

This project uses the public HC3 dataset from Hello-SimpleAI:

https://huggingface.co/datasets/Hello-SimpleAI/HC3

The experiment uses the `open_qa` split. Human answers are labelled as `0`, and ChatGPT answers are labelled as `1`.

The working subset used in the experiment contains:

| Metric | Value |
|---|---:|
| Samples | 2000 |
| Human samples | 1000 |
| AI samples | 1000 |
| Mean words | 74.23 |
| Median words | 54.00 |

## Project Structure

```text
ai-text-detection-stylometry-hc3/
├── ai_text_detection_hc3.ipynb
├── model_comparison.csv
├── dataset_profile.csv
├── stylometric_ablation.png
├── model_comparison.png
├── best_confusion_matrix.png
└── misclassified_examples.csv
└── README.md
