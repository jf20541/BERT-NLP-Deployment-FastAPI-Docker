from transformers import BertTokenizer
import torch

TRAINING_FILE = "../inputs/train.csv"
TRAINING_FILE_CLEAN_FOLDS = "../inputs/train_clean_folds.csv"
BERT_PATH = "bert_based_uncased"
MODEL_PATH = "../models/model.bin"
TOKEN = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MAX_LEN = 64
EPOCHS = 2
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4
DEVICE = torch.device("cuda")
