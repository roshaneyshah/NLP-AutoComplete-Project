# NLP Auto-Complete Project

Build an auto-complete system using N-gram language models trained on Twitter data.

## Overview

This project implements a complete NLP pipeline:
- **Part A:** Data preprocessing (tokenization, vocabulary, OOV handling)
- **Part B:** N-gram language models (1-gram to 6-gram)
- **Part C:** Model evaluation (perplexity scores)
- **Part D:** Auto-complete system (word suggestions)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

Or in Google Colab:
```python
exec(open('SUBMISSION.py').read())
```

## Project Structure

```
NLP-AutoComplete-Project/
├── SUBMISSION.py              # Main code
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── training.1600000.processed.noemoticon.csv
│   └── processed/
└── src/
    ├── part_a_preprocessing/
    ├── part_b_language_model/
    ├── part_c_evaluation/
    └── part_d_autocomplete/
```

## Dataset

Sentiment140: 1.6M tweets
- Download: https://www.kaggle.com/datasets/kazanova/sentiment140
- Format: CSV with text in column 5

## Results

Typical output:
```
Train: 40,000 | Test: 10,000 | Vocab: 5,901

1-gram perplexity: 319.59
2-gram perplexity: 289.55
3-gram perplexity: 1256.35

'i love' → ['you', 'to', 'the', 'it', 'my']
'how are you' → ["'re", 'have', '!', 'are', '.']
```

## Authors

Roshane Shahbaz (220444)

## Deadline

May 17, 2026, 11:59 PM
