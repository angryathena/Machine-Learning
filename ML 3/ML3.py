import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from IPython.display import display


# This method generates the scatter plot
def scatterPlot(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Labeling axes and setting the ticks
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    plt.xticks([-2, -1, 0, 1, 2])
    plt.yticks([-2, -1, 0, 1, 2])
    ax.set_zticks([-1, 0, 1, 2])
    # Plotting
    scatter = ax.scatter3D(X[:, 0], X[:, 1], y, c=(y),
                           cmap=plt.get_cmap('summer'), alpha=1)
    # Adding a legend
    cbarS = plt.colorbar(scatter)
    cbarS.set_label('Data points')
    plt.show()


# This method trains the classifier, keeps track of parameters, and plots predictions
def classify(X, y, CRange, clf):
    # I am saving parameters to a dataframe to create coefficient tables
    Xdata = pd.DataFrame({
        'X1': X[:, 1],
        'X2': X[:, 2]})
    dfPoly = pd.DataFrame()
    for i, C in enumerate(CRange):
        # Training model
        clf.set_params(alpha=1 / (2 * C))
        clf.fit(X, y)
        # Saving coefficients
        d = dict(enumerate(clf.coef_.flatten(), 1))
        dfPoly = dfPoly.append(pd.DataFrame(d, index=['C=' + repr(C)]))
        # Saving intercept
        dfPoly.iloc[i, 0] = clf.intercept_
        # Plotting the prediction surface and data points
        Xtest = []
        for i in range(-20, 20, 5):
            for j in range(-20, 20, 5):
                Xtest.append([i / 10, j / 10])
        Xtest = np.array(Xtest)
        XtestPoly = poly.fit_transform(Xtest)
        yTest = clf.predict(XtestPoly)
        fig = plt.figure()
        plt.rcParams['font.size'] = '16'
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter3D(X[:, 1], X[:, 2], y, c=(y),
                               cmap=plt.get_cmap('summer'), alpha=1)
        surface = ax.plot_trisurf(Xtest[:, 0], Xtest[:, 1], yTest,
                                  cmap=plt.get_cmap('spring'), alpha=0.42,
                                  edgecolor='black', linewidth=0.2)
        # Adding legends
        cbarS = plt.colorbar(scatter)
        cbarF = plt.colorbar(surface)
        cbarS.set_label('Data points')
        cbarF.set_label('Prediction')
        # Setting labels and ticks
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('y')
        plt.xticks([-2, -1, 0, 1, 2])
        plt.yticks([-2, -1, 0, 1, 2])
    plt.show()
    # Naming columns as their coresponding polynomial features and displaying the parameter dataframe
    dfPoly.columns = poly.get_feature_names_out(Xdata.columns)
    display(dfPoly)


# This method plots MSE for various Cs to select the best hyperparameter
def crossValidation(X, y, step, clf):
    mean_error = []
    std_error = []
    CRange = [x * step for x in range(1, 21)]
    for C in CRange:
        clf.set_params(alpha=1 / (2 * C))
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            clf.fit(X[train], y[train])
            yPred = clf.predict(X[test])
            temp.append(mean_squared_error(y[test], yPred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    fig = plt.figure()
    plt.rcParams['font.size'] = '16'
    plt.errorbar(CRange, mean_error, yerr=std_error, color='deeppink')
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.xlim(0, step * 20)
    plt.show()


# Reading data from the .csv file and processing it
df = pd.read_csv("data.csv", header=None)
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]
# Creating the polynomial features
poly = PolynomialFeatures(degree=5)
XPoly = poly.fit_transform(X)

# Uncomment the desired classifier
clf, CRange, step = Lasso(), [1, 10, 1000], 2.5
# clf,CRange,step  = Ridge(), [0.00003,0.03,30],0.01


scatterPlot(X, y)
classify(XPoly, y, CRange, clf)
crossValidation(XPoly, y, step, clf)
