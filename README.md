# BERT-NLP


## Objective
Predict a binary NLP sentiment classification for the IMDB dataset with 50,000 reviews with an evenly distributed target values **[1:Positive & 2:Negative]** using a **BERT (Bidrectional Representations Encoder Transformer**.Measure BERT's performance with **accuracy score** since the target values are evenly distributed. 

## Output 
```
                               ....                   ....                    ....

```

## Repository File Structure
    ├── src          
    │   ├── train.py              # Training BERT model and evaluating metric (accuracy)
    │   ├── model.py              # BERT Base architecture with 12 Layers, 768 hidden size, 12 self-attention heads
    │   ├── engine.py             # Class Engine for Training, Evaluation, and BCE Loss function 
    │   ├── dataset.py            # Custom Dataset that return a paris of [input_ids, targets, tokens, masks] as tensors
    │   ├── create_folds_clean.py # Remove unwanted characters and initiated Stratified 5-Folds cross-validator
    │   ├── googlecolab.py        # Set up VSCode on Google Colab 
    │   └── config.py             # Define path as global variable
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
```
GRU(
  (embedding): Embedding(180446, 100)
  (lstm): GRU(100, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
  (out): Linear(in_features=512, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
```  

## GPU's Menu Accelerator
```
Sat Aug 14 00:31:06 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.42.01    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   39C    P0    33W / 250W |   2047MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
