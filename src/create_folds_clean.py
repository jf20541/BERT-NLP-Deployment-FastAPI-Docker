import pandas as pd
from bs4 import BeautifulSoup
import re
import config
from sklearn.model_selection import StratifiedKFold


def remove_special_characters(text):
    """Remove unwanted characters [!@#$%^&*()]
    Args:
        text (string): description review
    Returns:
        [string]: cleaned reviews
    """
    soup = BeautifulSoup(text, "html.parser")
    review = soup.get_text()
    review = r"[^a-zA-z0-9\s]"
    review = re.sub(review, "", text)
    return review.lower()


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    df.sentiment = [1 if each == "positive" else 0 for each in df.sentiment]
    df["review"] = df["review"].apply(remove_special_characters)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    targets = df.sentiment.values
    kf = StratifiedKFold(n_splits=5)
    for f, (train, valid) in enumerate(kf.split(X=df, y=targets)):
        df.loc[valid, "kfold"] = f

    df.to_csv(config.TRAINING_FILE_CLEAN_FOLDS, index=False)
