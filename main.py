import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from wordcloud import WordCloud
import seaborn as sns
import re
from nltk.stem import snowball
from nltk.tokenize import word_tokenize
import treetaggerwrapper

dict_numbers = {
    '1': 'uno',
    '2': 'due',
    '3': 'tre',
    '4': 'quattro',
    '5': 'cinque',
    '10': 'dieci',
    '20': 'venti',
    '30': 'trenta',
    '40': 'quaranta',
    '50': 'cinquanta',
    '100': 'cento',
    '200': 'duecento',
    '300': 'trecento',
    '400': 'quattrocento',
    '500': 'cinquecento'
}


class Lemmatizer(object):
    def __init__(self):
        self.stemming = snowball.ItalianStemmer()
        self.tagger = treetaggerwrapper.TreeTagger(TAGLANG='it')

    def __call__(self, document):
        X_pre = []
        document = re.sub('[^A-Za-z0-9]+', ' ', document.lower())
        for word_token in word_tokenize(document):

            if word_token in dict_numbers:
                word_token = dict_numbers[word_token]

            if len(word_token) >= 2 and str.isalpha(word_token) is True:
                #print(word_clean)
                lemma = self.tagger.tag_text(word_token)[0]
                word = lemma.split("\t")[2]
                X_pre.append(word)

        return X_pre


def generate_wordclouds(X_tfidf, y, word_positions):
    top_count = 100
    for label in range(0, 2):
        tfidf = X_tfidf[y == label]
        tfidf_sum = np.sum(tfidf, axis=0)
        tfidf_sum = np.asarray(tfidf_sum).reshape(-1)
        top_indices = tfidf_sum.argsort()[-top_count:]
        term_weights = {word_positions[idx]: tfidf_sum[idx] for idx in top_indices}
        wc = WordCloud(max_font_size=1200, max_words=800, background_color="white")
        wordcloud = wc.generate_from_frequencies(term_weights)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.show()
    return


def grid(alg, par):
    grid_obj = GridSearchCV(alg, par, cv=5, n_jobs=1)
    grid_obj = grid_obj.fit(X_train_tfidf, y_train)
    clf = grid_obj.best_estimator_
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    print("Best estimator: ")
    print(clf)
    return y_pred


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def graph_comparison(acc_list, f1_list):
    bars = ['SVM', 'MultinomialNB', 'RandomForest', 'SGD']
    y_pos = np.arange(len(bars))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(y_pos - width/2, acc_list, width, label="Accuracy", color="lightblue")
    rects2 = ax.bar(y_pos + width/2, f1_list, width, label="F1-score", color="lightcoral")
    ax.set_ylim(0.9, 1)
    ax.set_ylabel("Measures")
    ax.set_title("Algorithm comparison")
    ax.set_xticks(y_pos)
    ax.legend()
    ax.set_xticklabels(bars)

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()
    plt.show()


def print_stat(y_test, y_pred):
    print('Accuracy of algorithm -->', f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    print("******")


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# **Reading the file**
print("Starting read dataset")
file_dir = 'dataset_winter_2020/development.csv'
dataset = pd.read_csv(file_dir, encoding='utf-8')
encoding = {"pos": 1, "neg": 0}
print(len(dataset))
print(dataset.count())
data = dataset['text']
labels = dataset['class'].replace(encoding)
sns.countplot(x='class', data=dataset)

# Bilanciamento classi
new_label = []
new_x = []

for x, label in zip(data, labels):
    if label == 0:
        new_label.append(label)
        new_x.append(x)

data = data.append(pd.Series(new_x))
labels = labels.append(pd.Series(new_label))

X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, test_size=0.2, shuffle=True)

lemmatizer = Lemmatizer()
sw = stopwords.words("italian")
sw.remove("non")
"""sw_list_st = ['abbi', 'abbiam', 'avev', 'avend', 'avess', 'avesser', 'avev', 'avrem', 'nostr', 'pag', 'perc', 'alcun',
            'altr', 'ancor', 'quand', 'dett', 'dop', 'quind', 'far', 'quell', 'quind', 'son', 'stat', 'port', 'part', 'ari', 'mattin',
           'quant', 'alcun', 'propr', 'ogni', 'po', 'stess', 'poi', 'altro', 'dat', 'acqua', 'pot']"""
sw_list_lem = ['chiedere', 'avere', 'dire', 'cos', 'cosa', 'dare', 'dovere', 'fare', 'fatto', 'perch', 'stanza', 'giorno', 'andare', 'altro',
               'dopo', 'quindi', 'quando', 'essere', 'abbastanza', 'stare', 'poi', 'persona', 'pi', 'pu', ' qui', 'ancora', 'po', 'stesso', 'volta']

sw.extend(sw_list_lem)

print('Tokenizer')
vectorizer = TfidfVectorizer(tokenizer=lemmatizer, stop_words=sw, analyzer='word', ngram_range=(1, 2), min_df=0.0005, max_df=0.15)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(len(vectorizer.get_feature_names()))
df_idf = pd.DataFrame(vectorizer.idf_, index=vectorizer.get_feature_names(), columns=['id_weights'])

# **Training**

print('Training phase')
parameters_SVC = {
    'kernel': ['linear', 'rbf'],
    'C': [0.001, 0.01, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 1, 10]
}

parameters_MB = {
    'alpha': np.linspace(0.5, 1.5, 6),
    'fit_prior': [True, False],
}

parameters_RF = {
    'n_estimators': [200, 400, 600]
}

parameters_SGD = {
    'tol': [1e-3, 1e-5],
    'alpha': [1e-6, 0.01]
}

# Grid search

dict_alg = {'svc': [SVC(), parameters_SVC], 'mnb': [MultinomialNB(), parameters_MB],
            'rf': [RandomForestClassifier(), parameters_RF], 'sgd': [SGDClassifier(), parameters_SGD]}

""""
acc_list = []
f1_list = []
for el in dict_alg.keys():
    y_pred = grid(dict_alg[el][0], dict_alg[el][1])
    accuracy = accuracy_score(y_test, y_pred)
    acc_list.append(round(accuracy, 3))
    print(f'Accuracy of {el}: {accuracy}')
    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred)
    f1_list.append(round(f1, 3))
    print(f'F-measure of {el}: {f1}')
    print("******")
    
    graph_comparison(acc_list, f1_list)
"""

SVC = SVC(kernel='rbf', C=10, gamma=1)
SVC.fit(X_train_tfidf, y_train)
y_pred = SVC.predict(X_test_tfidf)
print_stat(y_test, y_pred)

word_positions = {v: k for k, v in vectorizer.vocabulary_.items()}
generate_wordclouds(X_test_tfidf, y_pred, word_positions)

# ** Evaluation **
print("Evaluation")
file_dir = 'dataset_winter_2020/evaluation.csv'
dataset_label = pd.read_csv(file_dir, encoding='utf-8')
X_f = dataset_label['text']

vectorizer = TfidfVectorizer(tokenizer=lemmatizer, stop_words=sw, analyzer='word', ngram_range=(1, 2), min_df=0.0005, max_df=0.15)
X_train = vectorizer.fit_transform(data)
X_final = vectorizer.transform(X_f)

svc = svm.SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
            probability=False, random_state=None, shrinking=True, tol=0.001,
            verbose=False)

svc.fit(X_train, labels)
predictions = svc.predict(X_final)
word_positions = {v: k for k, v in vectorizer.vocabulary_.items()}
generate_wordclouds(X_final, predictions, word_positions)
dict_labels = {
    1: 'pos', 0: 'neg'
}

with open('result_lem.csv', 'w', encoding='UTF-8', newline='') as myFile:
    writer = csv.writer(myFile, quoting=csv.QUOTE_NONE, escapechar='', delimiter=' ')
    writer.writerow(['Id'+','+'Predicted'])

    for document, label in enumerate(predictions):
        y = dict_labels[label]
        row = [str(document)+','+str(y)]
        writer.writerow(row)
        row.clear()
