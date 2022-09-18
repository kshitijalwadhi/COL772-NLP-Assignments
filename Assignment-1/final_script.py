import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
import nltk
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import tokenize
from tqdm import tqdm


def check_deps():
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("punkt")
    return True


check_deps()

TRAIN_DATA_PATH = "Data/train.csv"
NUM_FOLDS = 5
FRACTION_DATA = 1

stopwords = nltk.corpus.stopwords.words("english")
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

df = pd.read_csv(TRAIN_DATA_PATH)
df = df.sample(frac=FRACTION_DATA, random_state=42)

professions = df["profession"].unique()


def augment_data(df, profession, target_num):
    profiles = df[df["profession"] == profession]["profile"].values
    if len(profiles) > target_num:
        return df
    all_sentences = []
    for profile in profiles:
        sentences = tokenize.sent_tokenize(profile)
        sentences = [sentence for sentence in sentences if len(sentence) > 10]
        all_sentences.extend(sentences)
    new_profiles = []
    np.random.shuffle(all_sentences)
    NUM_SENTENCES = 4
    for i in range(target_num - len(profiles)):
        new_profile = ".".join(np.random.choice(all_sentences, NUM_SENTENCES))
        new_profiles.append(new_profile)
    new_df = pd.DataFrame({"profile": new_profiles, "profession": profession})
    combined_df = pd.concat([df, new_df])
    return combined_df


def clean_data(df):
    profiles = df["profile"]
    cleaned_profiles = []
    for profile in tqdm(profiles):
        profile = re.sub(r"[^a-zA-Z .]", " ", profile)
        profile = profile.lower()
        profile = profile.split()
        profile = [stemmer.stem(word) for word in profile if not word in set(stopwords)]
        profile = [
            lemmatizer.lemmatize(word) for word in profile if not word in set(stopwords)
        ]
        profile = " ".join(profile)
        cleaned_profiles.append(profile)
    df["profile"] = cleaned_profiles
    return df


def remove_punctuation(df, column):
    df[column] = df[column].str.replace("[^\w\s]", "")
    return df


def sample_data(df):
    X, y = df.drop("profession", axis=1), df["profession"]
    over = RandomOverSampler()
    X_sampled, y_sampled = over.fit_resample(X, y)
    X_sampled["profession"] = y_sampled
    return X_sampled


def get_X_and_y(data_df):
    y = data_df["profession"]
    X_vec = vectorizer.transform(data_df["profile"])
    return X_vec, y


df = clean_data(df)

kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=False)

scores = []

for fold_idx in range(5):
    train_index, val_index = next(kf.split(df["profile"], df["profession"]))
    train_data = df.iloc[train_index]
    val_data = df.iloc[val_index]

    AUGMENT_DATA = True
    if AUGMENT_DATA:
        max_num = int(max(train_data["profession"].value_counts()) / 10)
        for profession in tqdm(professions):
            train_data = augment_data(train_data, profession, max_num)

    train_data = remove_punctuation(train_data, "profile")
    val_data = remove_punctuation(val_data, "profile")

    MAX_FEATURES = 50000
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=MAX_FEATURES, sublinear_tf=True
    )
    vectorizer.fit_transform(train_data["profile"])

    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
    X_train_vec, y_train = get_X_and_y(train_data)
    from sklearn.svm import LinearSVC

    modelSVC = LinearSVC(C=0.1, penalty="l2", dual=False, max_iter=10000).fit(
        X_train_vec, y_train
    )

    X_val_vec, y_val = get_X_and_y(val_data)
    y_pred = modelSVC.predict(X_val_vec)

    print(f"Checking on fold {str(fold_idx)}")
    print("Micro F1: ", metrics.f1_score(y_val, y_pred, average="micro"))
    print("Macro F1: ", metrics.f1_score(y_val, y_pred, average="macro"))
    print(
        "Average F1: ",
        (
            metrics.f1_score(y_val, y_pred, average="micro")
            + metrics.f1_score(y_val, y_pred, average="macro")
        )
        / 2,
    )

    scores.append(
        (
            metrics.f1_score(y_val, y_pred, average="micro")
            + metrics.f1_score(y_val, y_pred, average="macro")
        )
        / 2
    )

print("Average F1 of total: ", np.mean(scores))
