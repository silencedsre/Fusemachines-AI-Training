import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from config.config import DATA_PATH

# nltk.download('stopwords')
# nltk.download('wordnet')

def lowercase(text):
    return text.lower()

def decontraction(string):
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'t", " not", string)
    string_decontracted = re.sub(r"\'ve", " have", string)
    return string_decontracted

def remove_punctuation(text):
    exclist = string.punctuation + string.digits  # remove punctuations and digits
    table_ = str.maketrans('', '', exclist)
    string_removed_punctuation = text.translate(table_)
    return string_removed_punctuation

def remove_stopwords(string):
    stop_words = stopwords.words('english')
    tokens = string.split()
    tokens = [token for token in tokens if token not in stop_words]
    string_stopword_removed = " ".join(tokens)
    return string_stopword_removed


def lemmatize_words(string, pos):
    lemmatizer = WordNetLemmatizer()
    tokens = string.split()
    lemmatized_tokens = [lemmatizer.lemmatize(i, pos=pos) for i in tokens]
    string_lemmatized_words = " ".join(lemmatized_tokens)
    return string_lemmatized_words

def label_encoding(ser):
    encoder = LabelEncoder()
    encoder.fit(ser)
    y = encoder.transform(ser)
    return y


def preprocess_data(df):
    df = df.rename(columns={0: "#", 1: "emotion", 2: "review"})
    df = df.drop(["#"], axis=1)
    df['review'] = df['review'].apply(lowercase)
    df['review'] = df['review'].apply(decontraction)
    df['review'] = df['review'].apply(remove_punctuation)
    df['review'] = df['review'].apply(remove_stopwords)
    df['review'] = df['review'].apply(lemmatize_words, args=('v',))
    df['review'] = df['review'].apply(lemmatize_words, args=('n',))

    #label encoding on df['emotion']
    df["emotion"] = label_encoding(df['emotion'])

    return df

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, header=None)
    df = preprocess_data(df)
    print(df.head())