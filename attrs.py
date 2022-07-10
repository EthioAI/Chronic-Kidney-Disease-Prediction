import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def load_dataset(path):
    """
    Loads the dataset from the given path.
    """
    return pd.read_csv(path, index_col="id")


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


class DummyColumnTransform(BaseEstimator, TransformerMixin):
    '''
    Column that won't be transformed
    '''
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        OneHotEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        if self.columns is not None:
            return X[self.columns]
        else:
            return X

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

class DataSetCleaner(BaseEstimator, TransformerMixin):
    """
    Transforms the data set by removing columns that are not
    needed for the analysis and make it similar with the train set.
    And also removes any row that has nan value.
    """
    def __init__(self, columns_to_keep):
        self.columns_to_keep = columns_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[self.columns_to_keep]
        X = X.dropna(axis=0)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def analyze_model(model, X_test, y_test):
    """
    Analyzes the model's performance on the test set.
    """
    print(f"Score: {model.score(X_test, y_test)}")

    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")

    model_mse = mean_squared_error(y_test, predictions)
    model_rmse = np.sqrt(model_mse)
    print(f"Root mean squared error: {model_rmse}")

