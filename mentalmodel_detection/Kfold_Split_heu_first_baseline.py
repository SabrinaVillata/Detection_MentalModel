import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, naive_bayes
from sklearn.model_selection import KFold
from sklearn.svm import SVC

kf = KFold(n_splits=5, random_state=42, shuffle=True)

def logReg_classification(X, y):
    print(">>> Preparing Logistic Regression Model... \n")
    logReg = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
    # FSCORE
    fscore_avg_heu = []

    # PRECISION
    prec_avg_heu = []

    # RECALL
    rec_avg_heu = []

    # ACCURACY
    acc_avg_heu = []

    n_fold = 1
    for train, test in kf.split(X):
        print("Fold: " + str(n_fold), "\n")
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # LOGISTIC REGRESSION
        model = logReg.fit(X_train, y_train)
        y_pred = logReg.predict(X_test)

        print("> Precision: ", str(precision_score(y_test, y_pred, average="macro")))
        print("> Recall: ", str(recall_score(y_test, y_pred, average="macro")))
        print("> F1: ", str(f1_score(y_test, y_pred, average="macro")))
        print("> Accuracy ", str(accuracy_score(y_test, y_pred)), "\n")
        prec_avg_heu.append(precision_score(y_test, y_pred, average="macro"))
        rec_avg_heu.append(recall_score(y_test, y_pred, average="macro"))
        fscore_avg_heu.append(f1_score(y_test, y_pred, average="macro"))
        acc_avg_heu.append(accuracy_score(y_test, y_pred))

    print("LogReg precision: ", str(sum(prec_avg_heu) / len(prec_avg_heu)))
    print("LogReg recall: ", str(sum(rec_avg_heu) / len(rec_avg_heu)))
    print("LogReg f-1: ", str(sum(fscore_avg_heu) / len(fscore_avg_heu)))
    print("LogReg accuracy: ", str(sum(acc_avg_heu)/len(acc_avg_heu)))


def SVM_classification(X, y):
    print(">>> Preparing SVM Model...")
    SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    # FSCORE
    fscore_avg_heu = []

    # PRECISION
    prec_avg_heu = []

    # RECALL
    rec_avg_heu = []

    # ACCURACY
    acc_avg_heu = []

    n_fold = 1
    for train, test in kf.split(X):
        print("Fold: " + str(n_fold), "\n")
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # SVM
        model = SVM.fit(X_train, y_train)
        y_pred = SVM.predict(X_test)

        print("> Precision: ", str(precision_score(y_test, y_pred, average="macro")))
        print("> Recall: ", str(recall_score(y_test, y_pred, average="macro")))
        print("> F1: ", str(f1_score(y_test, y_pred, average="macro")))
        print("> Accuracy ", str(accuracy_score(y_test, y_pred)), "\n")
        prec_avg_heu.append(precision_score(y_test, y_pred, average="macro"))
        rec_avg_heu.append(recall_score(y_test, y_pred, average="macro"))
        fscore_avg_heu.append(f1_score(y_test, y_pred, average="macro"))
        acc_avg_heu.append(accuracy_score(y_test, y_pred))

    print("SVM precision: ", str(sum(prec_avg_heu) / len(prec_avg_heu)))
    print("SVM recall: ", str(sum(rec_avg_heu) / len(rec_avg_heu)))
    print("SVM f-1: ", str(sum(fscore_avg_heu) / len(fscore_avg_heu)))
    print("SVM accuracy: ", str(sum(acc_avg_heu) / len(acc_avg_heu)))


def NB_classification(X, y):
    print(">>> Preparing Naive Bayes Model...")
    Naive = naive_bayes.GaussianNB()
    # FSCORE
    fscore_avg_heu = []

    # PRECISION
    prec_avg_heu = []

    # RECALL
    rec_avg_heu = []

    # ACCURACY
    acc_avg_heu = []

    n_fold = 1
    for train, test in kf.split(X):
        print("Fold: " + str(n_fold), "\n")
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # Naive Bayes
        model = Naive.fit(X_train, y_train)
        y_pred = Naive.predict(X_test)

        print("> Precision: ", str(precision_score(y_test, y_pred, average="macro")))
        print("> Recall: ", str(recall_score(y_test, y_pred, average="macro")))
        print("> F1: ", str(f1_score(y_test, y_pred, average="macro")))
        print("> Accuracy ", str(accuracy_score(y_test, y_pred)), "\n")
        prec_avg_heu.append(precision_score(y_test, y_pred, average="macro"))
        rec_avg_heu.append(recall_score(y_test, y_pred, average="macro"))
        fscore_avg_heu.append(f1_score(y_test, y_pred, average="macro"))
        acc_avg_heu.append(accuracy_score(y_test, y_pred))

        n_fold = n_fold + 1

    print("NaiveBayes precision: ", str(sum(prec_avg_heu) / len(prec_avg_heu)))
    print("NaiveBayes recall: ", str(sum(rec_avg_heu) / len(rec_avg_heu)))
    print("NaiveBayes f-1: ", str(sum(fscore_avg_heu) / len(fscore_avg_heu)))
    print("NaiveBayes accuracy: ", str(sum(acc_avg_heu) / len(acc_avg_heu)))

if __name__ == "__main__":

    df = pd.read_csv('full.csv')

    # print(">>> Splitting dataset into training and test set...")
    # kf = KFold(n_splits=5, random_state=42, shuffle=True)

    X = df[["words", "sentences", "quest_mark", "whq", "imper_quest", "places_services",	"simple_quest",	"sensitive_data", "interjections", "conditional_vb"]]
    X = np.array(X)
    y = df["Model"]
    y = np.array(y)

    logReg_classification(X, y)
    # exit()

    SVM_classification(X, y)
    # exit()

    NB_classification(X, y)
    # exit()