import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sentence_transformers import SentenceTransformer
from sklearn import metrics, naive_bayes
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import os

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

sbert = SentenceTransformer('all-MiniLM-L6-v2')


def perform_kf_logreg(X, y):
    logReg = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    # FSCORE
    fscore_avg = []
    # PRECISION
    prec_avg = []
    # RECALL
    rec_avg = []
    # ACCURACY
    acc_avg = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    n_fold = 1
    for i, (train, test) in enumerate(kf.split(X)):
        print("Fold: " + str(n_fold), "\n")
        train_data = X[train]
        train_prad = y[train]

        logReg.fit(train_data, train_prad)
        test_prad = logReg.predict(X[test])

        # print(f"Fold {i}, accuracy {accuracy_score(y[test], test_prad)}")
        print("> Precision: ", str(precision_score(y[test], test_prad, average="macro")))
        print("> Recall: ", str(recall_score(y[test], test_prad, average="macro")))
        print("> F1: ", str(f1_score(y[test], test_prad, average="macro")))
        print("> Accuracy ", str(accuracy_score(y[test], test_prad)), "\n")
        prec_avg.append(precision_score(y[test], test_prad, average="macro"))
        rec_avg.append(recall_score(y[test], test_prad, average="macro"))
        fscore_avg.append(f1_score(y[test], test_prad, average="macro"))
        acc_avg.append(accuracy_score(y[test], test_prad))

        n_fold = n_fold +1

    print("LogReg precision: ", str(sum(prec_avg) / len(prec_avg)))
    print("LogReg recall: ", str(sum(rec_avg) / len(rec_avg)))
    print("LogReg f-1: ", str(sum(fscore_avg) / len(fscore_avg)))
    print("LogReg accuracy: ", str(sum(acc_avg)/len(acc_avg)))


def perform_kf_SVM(X, y):
    print(">>> Preparing SVM Model...")
    SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    # FSCORE
    fscore_avg = []
    # PRECISION
    prec_avg = []
    # RECALL
    rec_avg = []
    # ACCURACY
    acc_avg = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    n_fold = 1
    for i, (train, test) in enumerate(kf.split(X)):
        print("Fold: " + str(n_fold), "\n")
        train_data = X[train]
        train_prad = y[train]

        SVM.fit(train_data, train_prad)
        test_prad = SVM.predict(X[test])

        print(f"Fold {i}, accuracy {accuracy_score(y[test], test_prad)}")
        print("> Precision: ", str(precision_score(y[test], test_prad, average="macro")))
        print("> Recall: ", str(recall_score(y[test], test_prad, average="macro")))
        print("> F1: ", str(f1_score(y[test], test_prad, average="macro")))
        print("> Accuracy ", str(accuracy_score(y[test], test_prad)), "\n")
        prec_avg.append(precision_score(y[test], test_prad, average="macro"))
        rec_avg.append(recall_score(y[test], test_prad, average="macro"))
        fscore_avg.append(f1_score(y[test], test_prad, average="macro"))
        acc_avg.append(accuracy_score(y[test], test_prad))

        n_fold = n_fold + 1

    print("SVM precision: ", str(sum(prec_avg) / len(prec_avg)))
    print("SVM recall: ", str(sum(rec_avg) / len(rec_avg)))
    print("SVM f-1: ", str(sum(fscore_avg) / len(fscore_avg)))
    print("SVM accuracy: ", str(sum(acc_avg) / len(acc_avg)))


def perform_kf_NB(X, y):
    print(">>> Preparing Naive Bayes Model...")
    Naive = naive_bayes.GaussianNB()
    # FSCORE
    fscore_avg = []
    # PRECISION
    prec_avg = []
    # RECALL
    rec_avg = []
    # ACCURACY
    acc_avg = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    n_fold = 1
    for i, (train, test) in enumerate(kf.split(X)):
        print("Fold: " + str(n_fold), "\n")
        train_data = X[train]
        train_prad = y[train]

        Naive.fit(train_data, train_prad)
        test_prad = Naive.predict(X[test])

        print(f"Fold {i}, accuracy {accuracy_score(y[test], test_prad)}")
        print("> Precision: ", str(precision_score(y[test], test_prad, average="macro")))
        print("> Recall: ", str(recall_score(y[test], test_prad, average="macro")))
        print("> F1: ", str(f1_score(y[test], test_prad, average="macro")))
        print("> Accuracy ", str(accuracy_score(y[test], test_prad)), "\n")
        prec_avg.append(precision_score(y[test], test_prad, average="macro"))
        rec_avg.append(recall_score(y[test], test_prad, average="macro"))
        fscore_avg.append(f1_score(y[test], test_prad, average="macro"))
        acc_avg.append(accuracy_score(y[test], test_prad))

        n_fold = n_fold + 1

    print("Naive Bayes precision: ", str(sum(prec_avg) / len(prec_avg)))
    print("Naive Bayes recall: ", str(sum(rec_avg) / len(rec_avg)))
    print("Naive Bayes f-1: ", str(sum(fscore_avg) / len(fscore_avg)))
    print("Naive Bayes accuracy: ", str(sum(acc_avg) / len(acc_avg)))



def doit_with_tfidf(df):
    column_trans = ColumnTransformer([
        ('tfidf', TfidfVectorizer(), 'turn'),
        ('number_scaler', MinMaxScaler(),
         ["words", "sentences", "quest_mark", "whq", "imper_quest", "places_services", "simple_quest", "sensitive_data",
          "interjections", "conditional_vb"])
    ])
    X = column_trans.fit_transform(df).toarray()

    y = df.loc[:, 'Model'].values
    # print(f"shapes {X.shape}, {y.shape}")

    perform_kf_logreg(X, y)
    perform_kf_SVM(X, y)
    perform_kf_NB(X, y)


def doit_with_sbert(df):
    tmp = df['turn'].apply(lambda x: sbert.encode(x).flatten()).values
    data = [x for x in tmp]
    tmp_df = pd.DataFrame(data=data, columns=[_ for _ in range(len(data[0]))])  # 2141 x 384
    X_turn = tmp_df.values  # to np.append

    column_trans = ColumnTransformer([
        ('number_scaler', MinMaxScaler(),
         ["words", "sentences", "quest_mark", "whq", "imper_quest", "places_services", "simple_quest", "sensitive_data",
          "interjections", "conditional_vb"])
    ])

    y = df.loc[:, 'Model'].values
    X = column_trans.fit_transform(df)
    # print(X.shape, type(X))
    X = np.hstack([X, X_turn])
    # print(X.shape)

    perform_kf_logreg(X, y)
    perform_kf_SVM(X, y)
    perform_kf_NB(X, y)


if __name__ == "__main__":

    df = pd.read_csv('full.csv')
    print(f"from file: {df.shape}")


    df = df.drop(columns=["dialogue_id"], axis=1)
    print(">>> Performing TFIDF vectorization...")
    doit_with_tfidf(df)

    print(">>> Performing SBERT vectorization...")
    doit_with_sbert(df)

