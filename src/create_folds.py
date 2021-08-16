import config
import pandas as pd
from sklearn.model_selection import StratifiedKFold



if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_FILE_CLEAN)
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    targets = df.sentiment.values 
    kf = StratifiedKFold(n_splits=5)
    for f, (train, valid) in enumerate(kf.split(X=df, y=targets)):
        df.loc[valid, 'kfold'] = f
    
    df.to_csv(config.TRAINING_FILE_FOLDS, index=False)