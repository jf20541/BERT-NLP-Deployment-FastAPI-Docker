import transformers
import torch
import torch.nn as nn
import config


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        o1, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        drop = self.dropout(o2)
        return self.out(drop)
