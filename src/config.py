from transformers import BertTokenizer

TRAINING_FILE = "../inputs/train.csv"
TRAINING_FILE_CLEAN_FOLDS = "../inputs/train_clean_folds.csv"
BERT_PATH = "bert_based_uncased"
MODEL_PATH = "../models/bert_model.bin"
TOKEN = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MAX_LEN = 64
EPOCHS = 2
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4
LEARNING_RATE = 0.00003
DROP_OUT = 0.25
