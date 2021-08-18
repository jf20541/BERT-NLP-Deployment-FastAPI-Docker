import torch.nn as nn
from transformers import BertModel
import config


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        # loading the model's weights from BERT path
        self.bert = BertModel.from_pretrained(config.BERT_PATH)
        # dropout of 30% (avoids overfitting)
        self.bert_drop = nn.Dropout(config.DROP_OUT)
        # BERT based model has a given 768 ouput features
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # o1 is sequence of hidden states for each token (MAX_LEN) vectors of size 768 for each batch
        # o2 is CLS token from the BERT pooler output
        o1, o2 = self.bert(ids, mask, token_type_ids, return_dict=False)
        # pass through dropout
        out = self.bert_drop(o2)
        # pass through linear layer
        return self.out(out)
