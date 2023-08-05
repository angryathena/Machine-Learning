import numpy as np
import pandas as pd

import emoji

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# nltk.download('stopwords')
# nltk.download('wordnet')

def matching():
    # Reading the listings csv and adding a column where reviews will be added
    listings = pd.read_csv('listings.csv')
    listings['reviews'] = ['" '] * len(listings)

    # Reading the reviews csv and dropping reviews for unlisted accommodations
    reviews = pd.read_csv('reviews.csv')
    reviews.drop(index=reviews[~reviews['listing_id'].isin(listings['id'].tolist())].index, inplace=True)
    reviews.reset_index(inplace=True, drop=True)

    # Matching each review to its accommodation
    for i, id in enumerate(reviews.loc[:, 'listing_id']):
        print(i)
        text = emoji.demojize(str(reviews.loc[i, 'comments']), delimiters=("", ""))
        listings.loc[listings['id'] == id, 'reviews'] = listings.loc[listings['id'] == id, 'reviews'].values[
                                                            0] + ' ' + text
    listings['reviews'] = listings['reviews'].astype(str) + ' "'

    # Dropping listings with missing data
    listings.dropna(inplace=True)
    listings.reset_index(inplace=True, drop=True)

    # Saving the resulting dataframe for later use
    listings.to_csv('listings+reviews.csv', index=False)


# matching()

# Reading the csv of listings and reviews
df = pd.read_csv('listings+reviews.csv')


def features(min, max):
    feature_names = []

    def lemmatizeVectorize(unprocessedReviews):
        # Removing stop words and lemmatizi ng
        lemmatizer = WordNetLemmatizer()
        reviews = []
        for review in unprocessedReviews:
            tokens = word_tokenize(review)
            words = [lemmatizer.lemmatize(token) for token in tokens]
            reviews.append(' '.join(words))

        # Vectorizing
        vectorizer = TfidfVectorizer(stop_words='english', min_df=min, max_df=max)
        reviews = vectorizer.fit_transform(reviews).toarray()
        feature_names.extend(vectorizer.get_feature_names_out())
        return reviews

    reviews = lemmatizeVectorize(df['reviews'])
    host = lemmatizeVectorize(df['host_about'])
    neighbourhood = lemmatizeVectorize(df['neighbourhood_cleansed'])
    neighbourhoodDescription = lemmatizeVectorize(df['neighborhood_overview'])
    listing = lemmatizeVectorize(df['description'])
    amenities = lemmatizeVectorize(df['amenities'])

    features = ['host_id', 'response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
                'host_listings_count',
                'host_total_listings_count', 'email', 'phone', 'work_email', 'host_has_profile_pic',
                'host_identity_verified', 'latitude', 'longitude', 'accommodates', 'shared_bathroom', 'bathrooms',
                'bedrooms', 'beds', 'price', 'minimum_nights', 'maximum_nights', 'has_availability', 'availability_30',
                'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'number_of_reviews_ltm',
                'number_of_reviews_l30d', 'ins1an1_bookable', 'calculated_host_listings_count', 'reviews_per_month'
                ]
    feature_names.extend(features)

    # Returning the complete list of features and their names
    return np.hstack([reviews, host, neighbourhood, neighbourhoodDescription, listing, amenities,
                      df[features].values]), feature_names


pairsDF = [[0.0, 0.75], [0.0, 1.0], [0.25, 1.0]]


def LRcrossvalidation(Xlist, y):
    fig = plt.figure()
    plt.rcParams['font.size'] = '16'
    plt.xlabel('Number of features')
    plt.ylabel('Mean Squared Error')
    plt.title('Linear Regression')

    # Iterating through the set of feature sets
    for j in range(3):
        print('(' + str(min) + ',' + str(max) + ')')

        model = LinearRegression()
        kfold = KFold(n_splits=5)
        mean_MSE = []
        std_MSE = []
        kRange = [1, 10, 50, 100, 150, 200]

        # Iterating through a range of the number of features to be considered by the model
        for k in kRange:

            # Selecting the k best features
            X_best = SelectKBest(r_regression, k=k).fit_transform(Xlist[j], y)
            MSE = []

            # Using KFold cross validation for a Linear Regression
            for train, test in kfold.split(X_best):
                model.fit(X_best[train], y[train])
                yPred = model.predict(X_best[test])
                MSE.append(mean_squared_error(y[test], yPred))
            a = np.array(MSE).mean()
            mean_MSE.append(a)
            std_MSE.append(np.array(MSE).std())

        # Plotting the mean and std MSE for each k
        plt.errorbar(kRange, mean_MSE, yerr=std_MSE, label=('(' + str(pairsDF[j][0]) + ',' + str(pairsDF[j][1]) + ')'))

    # Adding a line corresponding to a dummy regressor which always predicts the mean value
    dummy = DummyRegressor()
    dummy.fit(X, y)
    yPred = dummy.predict(X)
    dummyMSE = mean_squared_error(y, yPred)
    plt.axhline(y=dummyMSE, color='pink', label='Dummy')
    plt.legend()

    plt.show()


def SVRcrossvalidation(Xlist, y):
    fig, axs = plt.subplots(3, 2)
    plt.rcParams['font.size'] = '10'
    plt.title('Support Vector Regression')
    kRange = [10, 50, 100, 150, 200, 250]

    # Iterating through a range of the number of features to be considered by the model
    for i, k in enumerate(kRange):
        ax = axs[i % 3, i // 3]
        ax.set_xlabel('C')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title(f"K = {k}")
        print(k)
        # Iterating through the set of feature sets created earlier
        for j in range(3):
            print(pairsDF[j])
            X_best = SelectKBest(r_regression, k=k).fit_transform(Xlist[j], y)
            kfold = KFold(n_splits=5)
            mean_MSE = []
            std_MSE = []
            cRange = [0.000000001, 1, 5, 10, 15, 20]

            # Iterating through a list of C values
            for c in cRange:
                model = SVR(C=c)
                MSE = []

                # Using KFold cross validation for an SV Regression with a given C
                for train, test in kfold.split(X_best):
                    model.fit(X_best[train], y[train])
                    yPred = model.predict(X_best[test])
                    MSE.append(mean_squared_error(y[test], yPred))

                mean_MSE.append(np.array(MSE).mean())
                std_MSE.append(np.array(MSE).std())
            print(mean_MSE)
            ax.errorbar(cRange, mean_MSE, yerr=std_MSE,
                        label=('(' + str(pairsDF[j][0]) + ',' + str(pairsDF[j][1]) + ')'))

        # Adding a line corresponding to a dummy regressor which always predicts the mean value
        dummy = DummyRegressor()
        dummy.fit(X, y)
        yPred = dummy.predict(X)
        dummyMSE = mean_squared_error(y, yPred)
        ax.axhline(y=dummyMSE, color='pink', label='Dummy')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    plt.show()


# Making a list of features for each combination of min and max DF to avoid calling the features method more often than necessary
Xlist = []
featureList = []
for [minDF, maxDF] in pairsDF:
    X, feature_names = features(minDF, maxDF)
    Xlist.append(X)
    featureList.append(feature_names)
# List of review score columns
Ylist = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
         'review_scores_communication', 'review_scores_location', 'review_scores_value']
Ynames = ['Rating', 'Accuracy', 'Cleanliness', 'Check-in', 'Communication', 'Location', 'Value']

# Crossvalidation to select the best parameters for each target
# for Y in Ylist:
# LRcrossvalidation(Xlist,df[Y]) #10,0.0,0.75
# SVRcrossvalidation(Xlist,df[Y]) #100, 0.25,1.0,C=5

# Resulting sets of the best parameters for each combination of model and review score ty
bestParametersLR = [[50, 0], [10, 2], [50, 0], [50, 0], [50, 0], [10, 2], [10, 2]]
bestParametersSVR = [[100, 2, 5], [10, 2, 1], [50, 2, 1], [100, 2, 1], [50, 2, 1], [250, 1, 5], [50, 2, 1]]


# Bar plot of the 10 best LR features
def LRbestFreatures():
    for i, parameters in enumerate(bestParametersLR):
        # Get the features and their names corresponding to the best parameters for each target
        X_best = SelectKBest(r_regression, k=parameters[0]).fit_transform(Xlist[parameters[1]], df[Ylist[i]])
        bestFeatures = SelectKBest(r_regression, k=parameters[0]).fit(Xlist[parameters[1]], df[Ylist[i]]).get_support()
        print(sum(bestFeatures))
        bestFeaturesNames = [featureList[parameters[1]][j] for j in range(len(featureList[parameters[1]])) if
                             bestFeatures[j]]

        # Fit the model
        model = LinearRegression()
        model.fit(X_best, df[Ylist[i]])

        # Obtain the coeficients, record the original indices, and sort by absolute value
        coefs = model.coef_
        coefficientIndices = [(c, coefficient) for c, coefficient in enumerate(coefs)]
        sortedCoefficients = sorted(coefficientIndices, key=lambda x: abs(x[1]), reverse=True)

        # Get the names and coefficients of the 10 best features
        names = [bestFeaturesNames[c] for c, coefficient in sortedCoefficients[:10]]
        coefficients = [coefficient for c, coefficient in sortedCoefficients[:10]]

        # Print the review scores rating
        if i == 0:
            fig = plt.figure()
            plt.rcParams['font.size'] = '16'
            plt.title(Ynames[i])
            plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
            plt.barh(names, coefficients, color='pink')
            for i, v in enumerate(coefficients):
                plt.text(v + 0.01, i - 0.2, str(round(v, 2)), color='black')
            plt.xlim(0, max(coefficients) + 0.2)
            plt.show()
            fig, axs = plt.subplots(3, 2)
            plt.rcParams['font.size'] = '9'
        # Print the other review scores in 6 subplots
        else:
            ax = axs[(i - 1) % 3, (i - 1) // 3]
            ax.set_title(Ynames[i])
            ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
            ax.barh(names, coefficients, color='pink')
            ax.set_xlim(min(coefficients) - 0.01, max(coefficients) + 0.33)
            for i, v in enumerate(coefficients):
                ax.text(v + 0.01, i - 0.2, str(round(v, 2)), color='black')
    plt.show()


# Bar plot comparing LR, SVR and the baseline
def comparison():
    LRlist = []
    # Performances of the best LR
    for i, parameters in enumerate(bestParametersLR):
        X_best = SelectKBest(r_regression, k=parameters[0]).fit_transform(Xlist[parameters[1]], df[Ylist[i]])
        model = LinearRegression()
        model.fit(X_best, df[Ylist[i]])
        yPred = model.predict(X_best)
        LRlist.append(mean_squared_error(df[Ylist[i]], yPred))

    # Performances of the best SVR
    SVRlist = []
    for i, parameters in enumerate(bestParametersSVR):
        X_best = SelectKBest(r_regression, k=parameters[0]).fit_transform(Xlist[parameters[1]], df[Ylist[i]])
        model = SVR(C=parameters[2])
        model.fit(X_best, df[Ylist[i]])
        yPred = model.predict(X_best)
        SVRlist.append(mean_squared_error(df[Ylist[i]], yPred))

    # Performances of the baseline
    Dlist = []
    for y in Ylist:
        dummy = DummyRegressor()
        dummy.fit(Xlist[1], df[y])
        yPred = dummy.predict(Xlist[1])
        Dlist.append(mean_squared_error(df[y], yPred))

    # Bar plot of the performances for each target
    width = 0.2
    LRx = [i - width for i in range(1, 8)]
    SVRx = [i for i in range(1, 8)]
    Dx = [i + width for i in range(1, 8)]
    plt.bar(LRx, LRlist, width, label='LR')
    plt.bar(SVRx, SVRlist, width, label='SVR')
    plt.bar(Dx, Dlist, width, color='pink', label='Dummy')
    plt.xticks([i + width / 2 for i in range(1, 8)], Ynames)
    plt.legend()
    plt.show()

# LRbestFreatures
# comparison()
