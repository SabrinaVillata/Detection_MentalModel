import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn import metrics, naive_bayes
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

vectorizer = TfidfVectorizer()
sbert = SentenceTransformer('all-MiniLM-L6-v2')
kf = KFold(n_splits=5, random_state=42, shuffle=True)

def tfidf_vectorizer(X_train, X_test):
    # vectorizer.fit_transform(X_train)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

def sbert_vectorizer(X_train, X_test):
    X_train = sbert.encode(X_train)
    X_test = sbert.encode(X_test)
    return X_train, X_test


def logReg_classification(X, y):
    print(">>> Preparing Logistic Regression Model... \n")
    logReg = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
    # FSCORE
    fscore_avg_tfidf = []
    fscore_avg_sb = []

    # PRECISION
    prec_avg_tfidf = []
    prec_avg_sb = []

    # RECALL
    rec_avg_tfidf = []
    rec_avg_sb = []

    # ACCURACY
    acc_avg_tfidf = []
    acc_avg_sb = []

    n_fold = 1
    for train, test in kf.split(X):
        print("Fold: " + str(n_fold), "\n")
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        X_train_tfidf, X_test_tfidf = tfidf_vectorizer(X_train, X_test)


        # LOGISTIC REGRESSION TFIDF
        model = logReg.fit(X_train_tfidf, y_train)
        y_pred_tfidf = logReg.predict(X_test_tfidf)


        print(">>> TFIDF \n")
        print("> Precision: ", str(precision_score(y_test, y_pred_tfidf, average="macro")))
        print("> Recall: ", str(recall_score(y_test, y_pred_tfidf, average="macro")))
        print("> F1: ", str(f1_score(y_test, y_pred_tfidf, average="macro")))
        print("> Accuracy ", str(accuracy_score(y_test, y_pred_tfidf)), "\n")
        prec_avg_tfidf.append(precision_score(y_test, y_pred_tfidf, average="macro"))
        rec_avg_tfidf.append(recall_score(y_test, y_pred_tfidf, average="macro"))
        fscore_avg_tfidf.append(f1_score(y_test, y_pred_tfidf, average="macro"))
        acc_avg_tfidf.append(accuracy_score(y_test, y_pred_tfidf))


        X_train_sb, X_test_sb = sbert_vectorizer(X_train, X_test)
        print(X_train_sb)

        # LOGISTIC REGRESSION TFIDF
        model = logReg.fit(X_train_sb, y_train)
        y_pred_sb = logReg.predict(X_test_sb)

        print(">>> S-BERT \n")
        print("> Precision: ", str(precision_score(y_test, y_pred_sb, average="macro")))
        print("> Recall: ", str(recall_score(y_test, y_pred_sb, average="macro")))
        print("> F1: ", str(f1_score(y_test, y_pred_sb, average="macro")))
        print("> Accuracy ", str(accuracy_score(y_test, y_pred_sb)), "\n")
        prec_avg_sb.append(precision_score(y_test, y_pred_sb, average="macro"))
        rec_avg_sb.append(recall_score(y_test, y_pred_sb, average="macro"))
        fscore_avg_sb.append(f1_score(y_test, y_pred_sb, average="macro"))
        acc_avg_sb.append(accuracy_score(y_test, y_pred_sb))
        n_fold = n_fold + 1

    print("LogReg precision with tfidf: ", str(sum(prec_avg_tfidf) / len(prec_avg_tfidf)))
    print("LogReg recall with tfidf: ", str(sum(rec_avg_tfidf) / len(rec_avg_tfidf)))
    print("LogReg f-1 with tfidf: ", str(sum(fscore_avg_tfidf) / len(fscore_avg_tfidf)))
    print("LogReg accuracy with tfidf: ", str(sum(acc_avg_tfidf)/len(acc_avg_tfidf)))
    print()
    print("LogReg precision with sbert: ", str(sum(prec_avg_sb) / len(prec_avg_sb)))
    print("LogReg recall with sbert: ", str(sum(rec_avg_sb) / len(rec_avg_sb)))
    print("LogReg f-1 with sbert: ", str(sum(fscore_avg_sb) / len(fscore_avg_sb)))
    print("LogReg accuracy with sbert: ", str(sum(acc_avg_sb) / len(acc_avg_sb)))


def SVM_classification(X, y):
    print(">>> Preparing SVM Model...")
    SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

    # FSCORE
    fscore_avg_tfidf = []
    fscore_avg_sb = []

    # PRECISION
    prec_avg_tfidf = []
    prec_avg_sb = []

    # RECALL
    rec_avg_tfidf = []
    rec_avg_sb = []

    # ACCURACY
    acc_avg_tfidf = []
    acc_avg_sb = []


    n_fold = 1
    for train, test in kf.split(X):
        print("Fold: " + str(n_fold), "\n")
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        X_train_tfidf, X_test_tfidf = tfidf_vectorizer(X_train, X_test)

        # SVM TFIDF
        model = SVM.fit(X_train_tfidf, y_train)
        y_pred_tfidf = SVM.predict(X_test_tfidf)

        print(">>> TFIDF \n")
        print("> Precision: ", str(precision_score(y_test, y_pred_tfidf, average="macro")))
        print("> Recall: ", str(recall_score(y_test, y_pred_tfidf, average="macro")))
        print("> F1: ", str(f1_score(y_test, y_pred_tfidf, average="macro")))
        print("> Accuracy ", str(accuracy_score(y_test, y_pred_tfidf)), "\n")
        prec_avg_tfidf.append(precision_score(y_test, y_pred_tfidf, average="macro"))
        rec_avg_tfidf.append(recall_score(y_test, y_pred_tfidf, average="macro"))
        fscore_avg_tfidf.append(f1_score(y_test, y_pred_tfidf, average="macro"))
        acc_avg_tfidf.append(accuracy_score(y_test, y_pred_tfidf))


        X_train_sb, X_test_sb = sbert_vectorizer(X_train, X_test)

        # SVM
        model = SVM.fit(X_train_sb, y_train)
        y_pred_sb = SVM.predict(X_test_sb)

        print(">>> S-BERT \n")
        print("> Precision: ", str(precision_score(y_test, y_pred_sb, average="macro")))
        print("> Recall: ", str(recall_score(y_test, y_pred_sb, average="macro")))
        print("> F1: ", str(f1_score(y_test, y_pred_sb, average="macro")))
        print("> Accuracy ", str(accuracy_score(y_test, y_pred_sb)), "\n")
        prec_avg_sb.append(precision_score(y_test, y_pred_sb, average="macro"))
        rec_avg_sb.append(recall_score(y_test, y_pred_sb, average="macro"))
        fscore_avg_sb.append(f1_score(y_test, y_pred_sb, average="macro"))
        acc_avg_sb.append(accuracy_score(y_test, y_pred_sb))
        n_fold = n_fold + 1

    print("SVM precision with tfidf: ", str(sum(prec_avg_tfidf) / len(prec_avg_tfidf)))
    print("SVM recall with tfidf: ", str(sum(rec_avg_tfidf) / len(rec_avg_tfidf)))
    print("SVM f-1 with tfidf: ", str(sum(fscore_avg_tfidf) / len(fscore_avg_tfidf)))
    print("SVM accuracy with tfidf: ", str(sum(acc_avg_tfidf) / len(acc_avg_tfidf)))
    print()
    print("SVM precision with sbert: ", str(sum(prec_avg_sb) / len(prec_avg_sb)))
    print("SVM recall with sbert: ", str(sum(rec_avg_sb) / len(rec_avg_sb)))
    print("SVM f-1 with sbert: ", str(sum(fscore_avg_sb) / len(fscore_avg_sb)))
    print("SVM accuracy with sbert: ", str(sum(acc_avg_sb) / len(acc_avg_sb)))


def NB_classification(X, y):
    print(">>> Preparing Naive Bayes Model...")
    Naive_tf = naive_bayes.GaussianNB()
    Naive_sb = naive_bayes.GaussianNB()
    # FSCORE
    fscore_avg_tfidf = []
    fscore_avg_sb = []

    # PRECISION
    prec_avg_tfidf = []
    prec_avg_sb = []

    # RECALL
    rec_avg_tfidf = []
    rec_avg_sb = []

    # ACCURACY
    acc_avg_tfidf = []
    acc_avg_sb = []

    n_fold = 1
    for train, test in kf.split(X):
        print("Fold: " + str(n_fold), "\n")
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        X_train_tfidf, X_test_tfidf = tfidf_vectorizer(X_train, X_test)

        # Naive Bayes TFIDF
        model = Naive_tf.fit(X_train_tfidf.toarray(), y_train)
        y_pred_tfidf = Naive_tf.predict(X_test_tfidf.toarray())


        print(">>> TFIDF \n")
        print("> Precision: ", str(precision_score(y_test, y_pred_tfidf, average="macro")))
        print("> Recall: ", str(recall_score(y_test, y_pred_tfidf, average="macro")))
        print("> F1: ", str(f1_score(y_test, y_pred_tfidf, average="macro")))
        print("> Accuracy ", str(accuracy_score(y_test, y_pred_tfidf)), "\n")
        prec_avg_tfidf.append(precision_score(y_test, y_pred_tfidf, average="macro"))
        rec_avg_tfidf.append(recall_score(y_test, y_pred_tfidf, average="macro"))
        fscore_avg_tfidf.append(f1_score(y_test, y_pred_tfidf, average="macro"))
        acc_avg_tfidf.append(accuracy_score(y_test, y_pred_tfidf))


        X_train_sb, X_test_sb = sbert_vectorizer(X_train, X_test)


        print(X_train_sb.shape)

        # Naive Bayes SBERT
        model = Naive_sb.fit(X_train_sb, y_train)
        y_pred_sb = Naive_sb.predict(X_test_sb)

        print(">>> S-BERT \n")
        print("> Precision: ", str(precision_score(y_test, y_pred_sb, average="macro")))
        print("> Recall: ", str(recall_score(y_test, y_pred_sb, average="macro")))
        print("> F1: ", str(f1_score(y_test, y_pred_sb, average="macro")))
        print("> Accuracy ", str(accuracy_score(y_test, y_pred_sb)), "\n")
        prec_avg_sb.append(precision_score(y_test, y_pred_sb, average="macro"))
        rec_avg_sb.append(recall_score(y_test, y_pred_sb, average="macro"))
        fscore_avg_sb.append(f1_score(y_test, y_pred_sb, average="macro"))
        acc_avg_sb.append(accuracy_score(y_test, y_pred_sb))
        n_fold = n_fold + 1

    print("NaiveBayes precision with tfidf: ", str(sum(prec_avg_tfidf) / len(prec_avg_tfidf)))
    print("NaiveBayes recall with tfidf: ", str(sum(rec_avg_tfidf) / len(rec_avg_tfidf)))
    print("NaiveBayes f-1 with tfidf: ", str(sum(fscore_avg_tfidf) / len(fscore_avg_tfidf)))
    print("NaiveBayes accuracy with tfidf: ", str(sum(acc_avg_tfidf) / len(acc_avg_tfidf)))
    print()
    print("NaiveBayes precision with sbert: ", str(sum(prec_avg_sb) / len(prec_avg_sb)))
    print("NaiveBayes recall with sbert: ", str(sum(rec_avg_sb) / len(rec_avg_sb)))
    print("NaiveBayes f-1 with sbert: ", str(sum(fscore_avg_sb) / len(fscore_avg_sb)))
    print("NaiveBayes accuracy with sbert: ", str(sum(acc_avg_sb) / len(acc_avg_sb)))



if __name__ == "__main__":

    df = pd.read_csv('full.csv')

    print(">>> Splitting dataset into training and test set...")
    kf = KFold(n_splits=5, random_state=42, shuffle=True)


    X = df["turn"]
    X = np.array(X)
    y = df["Model"]
    y = np.array(y)

    logReg_classification(X, y)
    # exit()

    SVM_classification(X, y)
    # exit()

    NB_classification(X, y)
    # exit()