import pandas as pd
import numpy as np
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

import config
from engine import Engine
from model import BERT
from dataset import IMDBDataset

# device = torch.device('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(fold):
    df = pd.read_csv(config.TRAINING_FILE_CLEAN_FOLDS)[:11000]

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    y_train = train_df[["sentiment"]].values
    x_train = train_df.drop("sentiment", axis=1).values

    y_test = valid_df[["sentiment"]].values
    x_test = valid_df.drop("sentiment", axis=1).values

    train_dataset = IMDBDataset(x_train, y_train)
    test_dataset = IMDBDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE
    )

    model = BERT()
    model.to(device)

    # num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    num_train_steps = int(len(train_df) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(model.parameters(), lr=0.0001)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    eng = Engine(optimizer, model, device)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        eng.train_fn(train_loader, scheduler)
        eval_targets, eval_outputs = eng.eval_fn(test_loader)

        eval_outputs = np.array(eval_outputs) >= 0.5

        acccuracy = accuracy_score(eval_targets, eval_outputs)
        print(f"Accuracy Score {acccuracy}")
        if acccuracy > best_accuracy:
            torch.save(model.save_dict(), config.MODEL_PATH)
            best_accuracy = acccuracy


if __name__ == "__main__":
    train(fold=0)
