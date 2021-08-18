# BERT-NLP-Deployment


## Objective
Predict a binary NLP sentiment classification for the IMDB dataset with 50,000 reviews with an evenly distributed target values **[1:Positive & 2:Negative]** using a **BERT (Bidrectional Representations Encoder Transformer)**.Measure BERT's performance with **accuracy score** since the target values are evenly distributed. 


## Two BERT Phases
BERT is a bidirectional pre-trained transformer learning simultaneously on masked language modeling (MLM) objective and next sentence prediction (NSP). With bidirectional, it has a deeper sense of learning because it learns from left to right (vice-versa)

**Pre-Training (understands language and context) & Finetuning (how to solve the problem)**\
Learns Language by training two unsupervised tasks simultaneously, Mask Language Modeling (MLM) and Next Sentence Prediction (NSP)
- MLM: it randomly masks out 15% of the words in the input and replaces them with [MASK] tokens
- NSP:  the model gets pairs of sentences [SEP] tokens and learns to predict if the second sentence is random or not (binary classification problem) 

**Summary for pre-training BERT:**\
The input is a pair of sentences [SEP] with 15% of the tokens are [MASK]. It converts each token into a pre-trained word-embedding with three vectors (token, segment, and position embeddings). It outputs forward vectors for MLM and binary values for NSP. 

## Output 
**BERT Base Model**
```
100%|██████████| 5000/5000 
100%|██████████| 2500/2500 
Fold:0 Epoch:1/5, Train Accuracy: 80.22%, Eval Accuracy: 85.38%
....                               ....                  ....
100%|██████████| 5000/5000
100%|██████████| 2500/2500 
Fold:0 Epoch:3/5, Train Accuracy: 92.15%, Eval Accuracy: 89.02%
....                               ....                  ....
100%|██████████| 5000/5000
100%|██████████| 2500/2500
Fold:0 Epoch:5/5, Train Accuracy: 99.32%, Eval Accuracy: 94.15%
....                               ....                  ....
100%|██████████| 5000/5000
100%|██████████| 2500/2500
```
**Logistic Regression with Bag of Words**
```
Fold: 0, Accuracy = 88.49%
Fold: 1, Accuracy = 89.09%
Fold: 2, Accuracy = 88.38%
Fold: 3, Accuracy = 89.53%
Fold: 4, Accuracy = 89.31%
Accuracy Mean = 88.96%

```
**Multinomial Naive Bayes with Bag of Words**
```
Fold: 0, Accuracy = 84.81%
Fold: 1, Accuracy = 84.81%
Fold: 2, Accuracy = 84.51%
Fold: 3, Accuracy = 84.96%
Fold: 4, Accuracy = 84.80%
Accuracy Mean = 84.77%
```

## Repository File Structure
    ├── src          
    │   ├── train.py              # Training BERT model and evaluating metric (accuracy)
    │   ├── model.py              # BERT Base architecture with 12 Layers, 768 hidden size, 12 self-attention heads
    │   ├── engine.py             # Class Engine for Training, Evaluation, and BCE Loss function 
    │   ├── dataset.py            # Custom Dataset that return a paris of [input_ids, targets, tokens, masks] as tensors
    │   ├── create_folds_clean.py # Removed unwanted characters and initiated Stratified 5-Folds cross-validator
    │   ├── logistic_reg.py       # Logistic Regression with Bag of Words and evaluated metric
    │   ├── multi_naivebayes.py   # Naive Bayes Classifier for multinomial models with Bag of Words and evaluated metric
    │   ├── googlecolab.py        # Set up VSCode on Google Colab 
    │   └── config.py             # Defined path as global variable
    ├── inputs
    │   ├── train_clean_folds.csv # Cleaned Data and Stratified 5-Folds dataset
    │   └── train.csv             # Kaggle IMDB Dataset 
    ├── models
    │   ├── config.json           # BERT based defined hyperparameters
    │   ├── pytorch_model.bin.    # BERT pre-trained weights
    │   ├── vocab.txt             # Pretrained vocab files map
    │   └── bert_model.bin        # IMDB BERT's parameters saved into bert_model.bin 
    ├── requierments.txt          # Packages used for project
    └── README.md

## Model's Architecture
BERT Base has 12 Layers, 768 hidden size, 12 self-attention heads.\
The model below shows an example of **1-layer**
```
BERT(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (bert_drop): Dropout(p=0.3, inplace=False)
  (out): Linear(in_features=768, out_features=1, bias=True)
)
```  
