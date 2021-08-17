import pandas as pd
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from engine import Engine
from model import BERT
from dataset import IMDBDataset
import config


def train(fold):
    df = pd.read_csv(config.TRAINING_FILE_CLEAN_FOLDS)

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    y_train = train_df[["sentiment"]].values
    x_train = train_df.drop(["sentiment", "kfold"], axis=1).values

    y_test = valid_df[["sentiment"]].values
    x_test = valid_df.drop(["sentiment", "kfold"], axis=1).values

    train_dataset = IMDBDataset(x_train, y_train)
    test_dataset = IMDBDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE
    )

    model = BERT()
    model.to(config.DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(x_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    eng = Engine(optimizer, model, config.DEVICE)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        train_targets, train_outputs = eng.train_fn(train_loader, scheduler)
        eval_targets, eval_outputs = eng.eval_fn(test_loader)

        train_outputs = np.array(train_outputs) >= 0.5
        eval_outputs = np.array(eval_outputs) >= 0.5

        train_acc = accuracy_score(train_targets, train_outputs)
        eval_acc = accuracy_score(eval_targets, eval_outputs)
        print(
            f"Epoch:{epoch+1}/{config.EPOCHS}, Train Accuracy: {train_acc:.2f}%, Eval Accuracy: {eval_acc:.2f}%"
        )

        if eval_acc > best_accuracy:
            torch.save(model.save_dict(), config.MODEL_PATH)
            best_accuracy = eval_acc


if __name__ == "__main__":
    for i in range(5):
        train(i)
