from transformers import BertTokenizer

TRAINING_FILE = "../inputs/train.csv"
TRAINING_FILE_CLEAN_FOLDS = "../inputs/train_clean_folds.csv"
BERT_PATH = "../inputs/bert_base_uncased"
MODEL_PATH = "../models/model.bin"
TOKEN = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MAX_LEN = 512
EPOCHS = 5
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32
