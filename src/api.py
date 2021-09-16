from fastapi import FastAPI
import torch
import torch.nn as nn
import config
from transformers import BertModel, logging
from dataset import IMDBDataset

logging.set_verbosity_warning()
logging.set_verbosity_error()

app = FastAPI()

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(config.DROP_OUT)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, mask, token_type_ids, return_dict=False)
        out = self.bert_drop(o2)
        out = self.out(out)
        return torch.sigmoid(out)

model = BERT()
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))

@app.get("/predict")
def fetch_prediction(text: str):
    dataset = IMDBDataset([text], [-1])
    prediction = float(list(model.predict(dataset, batch_size=2))[0][0][0])
    return {"sentence": text, "positive": prediction, "negative": 1 - prediction}

# uvicorn api:app --host 0.0.0.0 --port 12000 --reload