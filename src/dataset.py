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
        reviews = str(self.reviews[idx])
        reviews = " ".join(reviews.split())

        inputs = self.tokenizer.encode_plus(
            reviews,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )
        return {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "sentiment": torch.tensor(self.sentiment[idx], dtype=torch.float),
        }
