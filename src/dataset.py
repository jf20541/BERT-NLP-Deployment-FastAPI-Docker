import torch
import config


class IMDBDataset:
    def __init__(self, reviews, sentiment):
        self.reviews = reviews
        self.sentiment = sentiment
        self.tokenizer = config.TOKEN
        self.max_len = config.MAX_LEN

    def __len__(self):
        return self.reviews.shape[0]

    def __getitem__(self, idx):
        reviews = str(self.reviews)
        reviews = " ".join(reviews.split())

        inputs = self.tokenizer.encode_plus(
            reviews,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "sentiment": torch.tensor(self.sentiment[idx], dtype=torch.float),
        }
