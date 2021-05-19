from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class Winsorizer(BaseEstimator, TransformerMixin):
    
    '''
    
    Winsorization transformer, clip the data based on quantiles
    
    Parameters:
    ------------------
    qmin: float, default=0.01
        lower quantile, any element smaller than this quantile will be clipped
        
    qmax: float, default=0.99
        upper quantile, any element larger than this quantile will be clipped
    
    '''
    
    def __init__(self, qmin=0.01, qmax=0.99):
        self.qmin = qmin
        self.qmax = qmax
    
    def fit(self, X, y=None):
        
        n, n_features = X.shape
        
        self.xmin = []
        self.xmax = []
        
        for i in range(n_features):
            self.xmin.append(np.quantile(X[:,i], self.qmin))
            self.xmax.append(np.quantile(X[:,i], self.qmax))
        
        
        return self
    
    def transform(self, X):
        
        n, n_features = X.shape
        
        for i in range(n_features):
            X[:,i] = np.clip(X[:,i], a_min=self.xmin[i], a_max=self.xmax[i])
        
        return X