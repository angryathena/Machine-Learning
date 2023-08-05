import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, \
    accuracy_score, f1_score, recall_score, precision_score, plot_confusion_matrix, jaccard_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve


def jacard_score(param, yPred):
    pass


def lrCrossValidation(X, y, cRange, dRange, colours):
    fig = plt.figure()
    plt.rcParams['font.size'] = '16'
    plt.xlabel('C')
    plt.ylabel('F1 score')
    for i, d in enumerate(list(dRange)):
        mean_accuracy = []
        std_accuracy = []
        for c in cRange:
            poly = PolynomialFeatures(degree=d)
            XPoly = poly.fit_transform(X)
            clf = LogisticRegression(penalty='l2', C=c, max_iter=1000)
            temp = []
            kf = KFold(n_splits=10)
            for train, test in kf.split(XPoly):
                clf.fit(XPoly[train], y[train])
                yPred = clf.predict(XPoly[test])
                temp.append(jaccard_score(y[test], yPred))
            mean_accuracy.append(np.array(temp).mean())
            std_accuracy.append(np.array(temp).std())
        plt.errorbar(cRange, mean_accuracy, yerr=std_accuracy, color=colours[i],
                     label=(str(d) + ' degree(s)'))
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()


def logisticRegression(X, y, C, d):
    poly = PolynomialFeatures(degree=d)
    XPoly = poly.fit_transform(X)
    clf = LogisticRegression(penalty='l2', C=C, max_iter=1000)
    clf.fit(XPoly, y)

    plt.rcParams['font.size'] = '16'

    yPred = clf.predict(XPoly)
    print('F-measure', f1_score(y, yPred, average='weighted'))
    print('Accuracy', accuracy_score(y, yPred))
    plt.scatter(X1[yPred == -1], X2[yPred == -1], marker='s', color='lightcoral', s=60)
    plt.scatter(X1[yPred == 1], X2[yPred == 1], marker='D', color='khaki', s=60)
    plt.scatter(X1[y == -1], X2[y == -1], marker='x', color='pink', s=20)
    plt.scatter(X1[y == 1], X2[y == 1], marker='+', color='olive', s=30)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(["$\^y$ = -1", "$\^y$ = 1", "y = -1", "y = 1"],
               bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()
    return [yPred, clf.decision_function(XPoly)]


def knnCrossValidation(X, y, kRange):
    fig = plt.figure()
    plt.rcParams['font.size'] = '16'
    plt.xlabel('K')
    plt.ylabel('F1 score')
    mean_accuracy = []
    std_accuracy = []
    for k in kRange:
        clf = KNeighborsClassifier(n_neighbors=k)
        temp = []
        kf = KFold(n_splits=10)
        for train, test in kf.split(X):
            clf.fit(X[train], y[train])
            yPred = clf.predict(X[test])
            temp.append(f1_score(y[test], yPred))
        mean_accuracy.append(np.array(temp).mean())
        std_accuracy.append(np.array(temp).std())
    plt.errorbar(kRange, mean_accuracy, yerr=std_accuracy, color='hotpink')
    plt.show()


def kNearestNeighbours(X, y, k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    yPred = clf.predict(X)
    plt.rcParams['font.size'] = '16'
    print('F-measure', f1_score(y, yPred, average='weighted'))
    print('Accuracy', accuracy_score(y, yPred))
    plt.scatter(X1[yPred == -1], X2[yPred == -1], marker='s', color='lightcoral', s=60)
    plt.scatter(X1[yPred == 1], X2[yPred == 1], marker='D', color='khaki', s=60)
    plt.scatter(X1[y == -1], X2[y == -1], marker='x', color='pink', s=20)
    plt.scatter(X1[y == 1], X2[y == 1], marker='+', color='olive', s=30)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(["$\^y$ = -1", "$\^y$ = 1", "y = -1", "y = 1"],
               bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()
    return [yPred, clf.predict_proba(X)[:, 1]]


def confusionMatrix(y, yB, yLR, yKNN):
    print(accuracy_score(y, yB))
    print(accuracy_score(y, yLR))
    print(accuracy_score(y, yKNN))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    baseline = metrics.confusion_matrix(y, yB)
    lr = metrics.confusion_matrix(y, yLR)
    knn = metrics.confusion_matrix(y, yKNN)
    metrics.ConfusionMatrixDisplay(confusion_matrix=baseline, display_labels=[-1, 1])\
        .plot(cmap='spring',ax=ax1,colorbar=False)
    metrics.ConfusionMatrixDisplay(confusion_matrix=lr, display_labels=[-1, 1])\
        .plot(cmap='spring', ax=ax2,colorbar=False)
    metrics.ConfusionMatrixDisplay(confusion_matrix=knn, display_labels=[-1, 1])\
        .plot(cmap='spring', ax=ax3,colorbar=False)
    ax1.set_title('Baseline')
    ax2.set_title('Logistic Regression')
    ax3.set_title('K Nearest Neighbours')
    plt.show()

def roc(y, LR, KNN):
    plt.rcParams['font.size'] = '16'
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    LRfpr, LRtpr, _ = roc_curve(y, LR)
    KNNfpr, KNNtpr, _ = roc_curve(y, KNN)
    plt.plot(LRfpr, LRtpr, color='hotpink', label='Logistic Regression')
    plt.plot(KNNfpr, KNNtpr, color='pink', label='K Nearest Neighbours')

    plt.plot([0, 1], [0, 1], color='olive', linestyle='--', label='Baseline')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()


df = pd.read_csv("data.csv", header=None)
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

colours = ['pink', 'deeppink', 'violet', 'blueviolet', 'skyblue', 'royalblue']
cRange = [0.01, 0.1, 0.5, 1, 3, 5, 10]
dRange = [1, 2, 3, 5, 7, 9]
kRange = range(1, 30)
C, d, k = 1, 1, 7 #I chose 1, 1, 7 for dataset 1 and 0.5, 2, 15 for dataset 2

''' Uncomment the desired method. logisticRegression and kNearestNeighbours
 will display the prediction scatter plots and are used by other methods. 
 confusionMatrix and roc will display both scatter plots as well'''

lrCrossValidation(X,y,cRange,dRange,colours)
#knnCrossValidation(X,y,kRange)
#logisticRegression(X,y,C,d)
#kNearestNeighbours(X, y, k)
#confusionMatrix(y, pd.DataFrame([1] * len(y)).iloc[:, 0], logisticRegression(X, y, C, d)[0],kNearestNeighbours(X, y, k)[0])
#roc(y,logisticRegression(X,y,C,d)[1],kNearestNeighbours(X, y, k)[1])
