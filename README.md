# Toxic Comment Classification

Multilabel Text Classification with:
- TF-IDF + LogisiticRegression (baseline)
- Pretrained BertModel

## Dataset

The dataset is available at [kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data "Toxic Comment Classification Challenge").
It contains more than 310k Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

## Config

The user interface consists of file:
- ```config.yaml``` - general configuration with data and model parameters

Default **config.yaml**:
```
seed: 42

data:
  train_data_path: ../data/train.csv
  test_data_path: ../data/test.csv
  test_size: 0.05
  sep: ','
  text_column: comment_text
  target_columns: ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  output_path: output
  log_file_path: logs.txt

model_type: baseline  # bert (baseline = tf_idf + logreg)
save_model: true
test_run: true


tf-idf:
  word:
    sublinear_tf: true
    strip_accents: unicode
    analyzer: word
    ngram_range: (1, 1)
    max_features: 10000
  char:
    sublinear_tf: true
    strip_accents: unicode
    analyzer: char
    ngram_range: (2, 6)
    max_features: 50000

logreg:
  penalty: l2
  C: 1.0
  class_weight: balanced
  solver: lbfgs
  n_jobs: -1

bert:
  bert_model_name: bert-base-cased
  max_token_len: 128
  batch_size: 32
  epochs: 2
  learning_rate: 0.00002
  warmup_frac: 0.2
```

## Usage

Run in **terminal**: ```python -m Toxic_comment_classification```

## Output

After training the model, the pipeline will return the following files:

- ```logreg_model.joblib``` / ```bert_best_model.pt``` -- saved models 
- ```logs.txt``` -- file with logs

