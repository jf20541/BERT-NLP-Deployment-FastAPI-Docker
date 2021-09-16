from transformers import BertTokenizer
import torch

TRAINING_FILE = "../inputs/train.csv"
TRAINING_FILE_CLEAN_FOLDS = "../inputs/train_clean_folds.csv"
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "../models/bert_model.bin"
DEVICE = torch.device("cuda")
TOKEN = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MAX_LEN = 64
EPOCHS = 5
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4
LEARNING_RATE = 0.00003
DROP_OUT = 0.25