import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


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

age_i, bp_i, bgr_i, su_i, bu_i, al_i, = 0, 1, 5, 4, 6, 3

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    For attributes/columns combination creation
    It should be in a format of: [column_name]_per_[column_name]
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        bp_per_age = X.iloc[:, bp_i] / X.iloc[:, age_i]
        bgr_per_su = X.iloc[:, su_i] / X.iloc[:, bgr_i]
        bu_per_al = X.iloc[:, al_i] / X.iloc[:, bu_i]

        return np.c_[X, bp_per_age, bgr_per_su, bu_per_al]

def analyze_model(model, X_test, y_test):
    """
    Analyzes the model's performance on the test set.
    """
    y_pred = model.predict(X_test)

    conf_mx = confusion_matrix(y_test, y_pred)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()

    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('F1: %.3f' % f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

