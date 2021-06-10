'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        self.degree = degree
        self.reg_lambda = reg_lambda
        self.theta = None
        self.mean = None
        self.std = None

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        poly_feat_X = X
        for i in range(1, self.degree):
            poly_feat_X = np.c_[poly_feat_X, X**i]
        assert poly_feat_X.shape == (X.shape[0], self.degree)
        return poly_feat_X

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        n = len(X)

        X_ = self.polyfeatures(X, self.degree)

        # standardize
        self.mean = X_.mean(axis=0)
        self.std = X_.std(axis=0)
        X_ = (X_ - self.mean) / self.std
        X_ = np.c_[np.ones([n, 1]), X_]

        n, d = X_.shape

        reg_matrix = self.reg_lambda * np.eye(d)
        reg_matrix[0, 0] = 0

        self.theta = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        n, _ = X.shape
        X_ = self.polyfeatures(X, self.degree)
        X_ = (X_ - self.mean) / self.std
        X_ = np.c_[np.ones([n, 1]), X_]

        return X_.dot(self.theta)


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        reg_lambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape
    assert Xtrain.shape == Ytrain.shape
    assert Xtest.shape == Ytest.shape

    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
    for i in range(1, n):
        X = Xtrain[0:i+1]
        y = Ytrain[0:i+1]

        model.fit(X,y)

        ytrain = model.predict(X)
        ytest = model.predict(Xtest)

        errorTrain[i] = np.mean((ytrain-y)**2)
        errorTest[i] = np.mean((ytest-Ytest)**2)

    return errorTrain, errorTest
