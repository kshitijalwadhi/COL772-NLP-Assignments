import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import nltk
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tqdm import tqdm

TRAIN_DATA_PATH = "Data/train.csv"
NUM_FOLDS = 5
FRACTION_DATA = 1

stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def check_deps():
    nltk.download('stopwords')
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    return True

class DataLoader:
    def __init__(self, train_data_path, vectorizer_type = "tf_idf", ngram_range = (1,1)):
        self.df = pd.read_csv(train_data_path)
        self.df = self.df.sample(frac=FRACTION_DATA, random_state = 42)
        self.clean_data()
        if vectorizer_type == "tf_idf":
            self.vectorizer = TfidfVectorizer(ngram_range = ngram_range)
        elif vectorizer_type == "count":
            self.vectorizer = CountVectorizer(ngram_range = ngram_range)
        self.vectorizer.fit_transform(self.df["profile"])

    def sample_data(self, df):
        X, y = df.drop("profession", axis=1), df["profession"]
        over = RandomOverSampler()
        X_sampled, y_sampled = over.fit_resample(X, y)
        X_sampled["profession"] = y_sampled
        return X_sampled
    
    def clean_data(self):
        profiles = self.df["profile"]
        cleaned_profiles = []
        for profile in tqdm(profiles):
            profile = re.sub(r'[^\w\s]', ' ', profile)
            profile = profile.lower()
            profile = profile.split()
            profile = [stemmer.stem(word) for word in profile if not word in set(stopwords)]
            profile = [lemmatizer.lemmatize(word) for word in profile if not word in set(stopwords)]
            profile = ' '.join(profile)
            cleaned_profiles.append(profile)
        self.df["profile"] = cleaned_profiles

    def vectorize_data(self, data):
        return self.vectorizer.transform(data)
    
    def get_X_and_y(self,data_df):
        X = data_df.drop("profession", axis=1)
        y = data_df["profession"]
        X_vec = self.vectorize_data(data_df["profile"])
        return X_vec, y

if __name__=="__main__":

    check_deps()

    data_loader = DataLoader(train_data_path=TRAIN_DATA_PATH, vectorizer_type="tf_idf", ngram_range=(1,2))
    df = data_loader.df

    kf = KFold(n_splits=NUM_FOLDS, shuffle=False)
    ## Currently considering only one fold here
    train_index, val_index = next(kf.split(df))
    train_data = df.iloc[train_index]
    val_data = df.iloc[val_index]

    train_data = data_loader.sample_data(train_data)

    # shuffle data
    train_data = train_data.sample(frac=1, random_state = 42).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state = 42).reset_index(drop=True)

    X_train_vec, y_train = data_loader.get_X_and_y(train_data)

    print("[Log] Training the Classifier Model")
    # model = LogisticRegression().fit(X_train_vec, y_train)
    model = MultinomialNB().fit(X_train_vec, y_train)
    print("[Log] Training Complete")

    X_val_vec, y_val = data_loader.get_X_and_y(val_data)

    y_pred = model.predict(X_val_vec)
    
    print(metrics.accuracy_score(y_val, y_pred))
