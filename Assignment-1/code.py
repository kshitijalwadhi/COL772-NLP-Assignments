from distutils.command.clean import clean
from tabnanny import check
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import re
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

TRAIN_DATA_PATH = "Data/train.csv"
NUM_FOLDS = 5
FRACTION_DATA = 0.2

stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def check_deps():
    nltk.download('stopwords')
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    return True

class DataLoader:
    def __init__(self, train_data_path):
        self.df = pd.read_csv(train_data_path)
        self.vectorizer = TfidfVectorizer()
        self.sample_data()
        self.clean_data()
        _ = self.vectorizer.fit_transform(self.df["profile"])


    def sample_data(self):
        self.df = self.df.sample(frac=FRACTION_DATA)
        X,y = self.df.drop("profession", axis=1), self.df["profession"]
        over = RandomOverSampler()
        under = RandomUnderSampler()
        X_sampled, y_sampled = over.fit_resample(X, y)
        X_sampled["profession"] = y_sampled
        self.df = X_sampled
    
    def clean_data(self):
        profiles = self.df["profile"]
        cleaned_profiles = []
        for profile in tqdm(profiles):
            profile = re.sub(r'[^\w\s]', ' ', profile)
            profile = profile.lower()
            profile = profile.split()
            profile = [lemmatizer.lemmatize(word) for word in profile if not word in set(stopwords)]
            profile = ' '.join(profile)
            cleaned_profiles.append(profile)
        self.df["profile"] = cleaned_profiles

    def vectorize_data(self, data):
        return self.vectorizer.transform(data)

if __name__=="__main__":

    check_deps()

    data_loader = DataLoader(train_data_path=TRAIN_DATA_PATH)
    df = data_loader.df

    kf = KFold(n_splits=NUM_FOLDS)
    ## Currently considering only one fold here
    train_index, val_index = next(kf.split(df))
    train_data = df.iloc[train_index]
    val_data = df.iloc[val_index]

    # shuffle data
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    val_data = val_data.sample(frac=1).reset_index(drop=True)

    X_train = train_data.drop("profession", axis=1)
    y_train = train_data["profession"]

    X_train_vec = data_loader.vectorizer.transform(X_train["profile"])
    # X_train_vec = data_loader.vectorize_data(X_train["profile"])

    print("[Log] Training the Classifier Model")
    # model = LogisticRegression().fit(X_train_vec, y_train)
    model = MultinomialNB().fit(X_train_vec, y_train)
    print("[Log] Training Complete")

    X_val = val_data.drop("profession", axis=1)
    y_val = val_data["profession"]
    X_val_vec = vectorizer.transform(X_val["profile"])

    y_pred = model.predict(X_val_vec)
    
    print(metrics.accuracy_score(y_val, y_pred))
