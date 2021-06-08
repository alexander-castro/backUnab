import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import re, string, unicodedata
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

DIR = os.path.dirname(os.path.realpath('__file__'))
app = Flask(__name__)
CORS(app)

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def preprocessing(words):
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words

def stem_words(words):
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems + lemmas

with app.app_context():
    url = 'https://raw.githubusercontent.com/alexander-castro/test/main/artemis_dataset_release_v0.csv'
    data = pd.read_csv(url)
    data = data[data['emotion'] != 'something else']
    data = data[:20000]
    data['emotion'] = data['emotion'].map({'contentment': 'positive', 'awe': 'negative', 'sadness': 'negative', 'fear': 'negative', 'excitement': 'positive', 'amusement': 'positive', 'disgust': 'negative', 'anger': 'negative'})
    data = data.reset_index(drop=True)
    data['utterance'] = data['utterance'].apply(contractions.fix)
    data['utterance'] = data['utterance'].apply(word_tokenize).apply(preprocessing)
    data['utterance'] = data['utterance'].apply(stem_and_lemmatize)
    data['utterance'] = data['utterance'].apply(lambda x: ' '.join(map(str, x)))
    le = LabelEncoder()
    data['emotion'] = le.fit_transform(data.emotion.values)
    vectorizer = TfidfVectorizer()
    text = vectorizer.fit_transform(data.utterance)
    myData = pd.DataFrame(data=text.toarray(), columns=vectorizer.get_feature_names())
    text = []
    myData['-Emotion-'] = data['emotion']
    myData = myData.drop(myData.columns[myData.apply(lambda col: (col != 0).mean() < 0.001)], axis=1)
    y_train = pd.DataFrame(myData.pop('-Emotion-'))
    X_train = myData
    clf = LogisticRegression(max_iter=10000, solver='saga', penalty="elasticnet", l1_ratio=0.6)
    clf.fit(X_train,y_train.values.ravel())
    dump(clf, os.path.join(DIR,'model/dataModel.joblib'))

@app.route('/api/patients', methods=['POST'])
def postPatients():
    model = load(os.path.join(DIR,'model/dataModel.joblib'))
    clf = load(os.path.join(DIR,'model/clf.joblib'))
    cols = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']
    X = pd.DataFrame(data=request.json).transpose()
    X = pd.DataFrame(X, columns=cols)
    X = pd.DataFrame(column_trans_2.transform(X), columns=np.concatenate((categorical_columns, numeric_columns), axis=None), index=X.index)
    novilty = clf.predict(pd.DataFrame(X, columns=numeric_columns)).tolist()[0]
    X = pd.DataFrame(column_trans_1.transform(X))
    X.columns=np.concatenate((numeric_columns, categorical_columns), axis=None)
    result = {}
    result['response'] = model.predict(X.iloc[[0]]).tolist()[0]
    result['probability'] = model.predict_proba(X.iloc[[0]]).tolist()[0][1]
    result['novilty'] = novilty
    return jsonify({"Result": result})

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
    