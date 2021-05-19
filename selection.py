import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

class FittedTransformedTargetRegressor:
    
    '''
    helper class to ensemble two fitted transformers and estimators
    into a fitted class with score method implemented
    '''
    
    def __init__(self, preprocessor, regressor, func):
        
        self.preprocessor = preprocessor
        self.regressor = regressor
        self.func = func
        
    def score(self, X, y, sample_weight=None):
        
        X_preprocessed = self.preprocessor.transform(X)
        y_preprocessed = self.func(y)
        
        return self.regressor.score(X_preprocessed, y_preprocessed, sample_weight)
    
    def predict(self, X):
        
        '''
        re-implement the predict method
        '''
        
        X_preprocessed = self.preprocessor.transform(X)
        y_pred = self.regressor.predict(X_preprocessed)
        
        return y_pred
        

class SelectKBestIC(BaseEstimator, TransformerMixin):

    '''

    Select k features with the best Information Coefficient score 

    Parameters
    ---------------
    k: int
        select best k features, if k=-1 donot select

    Id: array-like
        array of Ids of the features for re-indexing purpose

    Date: array-like
        array of Dates of the features for re-indexing purpose

    feature_names: array-like
        list of feature names if you want to know which feature you've selected

    verbose: bool, default=False
        print the result of selection

    '''

    def __init__(self, k, Id=None, Date=None, feature_names=None, verbose=False):

        self.k = k
        self.Id = Id
        self.Date = Date
        self.feature_names = feature_names
        self.verbose = verbose

    def fit(self, X, y):
        '''
        for each feature, first groupby date
        then compute IC for each day
        finally compute the average
        '''

        if self.k >= X.shape[1] or self.k == -1:
            # selected number exceed the feature number
            # do nothing
            self.kept_idx = range(X.shape[1])
            return self
        # convert the numpy array into dataframe
        # so we can groupby
        data = {'Id': self.Id, 'Date': self.Date, 'y': y}
        for i in range(X.shape[1]):
            if self.feature_names is None:
                feature_name = 'f{:d}'.format(i)
            else:
                feature_name = self.feature_names[i]

            data[feature_name] = X[:, i]
        # now we have the dataframe
        df_X_y = pd.DataFrame(data)

        # for each feature, groupby Date and compute each day's IC
        self.IC = []
        for i in range(X.shape[1]):
            if self.feature_names is None:
                feature_name = 'f{:d}'.format(i)
            else:
                feature_name = self.feature_names[i]

            ics = []
            for date, df in df_X_y.groupby('Date'):
                ic = df[[feature_name, 'y']].corr().values[0, 1]
                if np.isnan(ic) == False:
                    ics.append(ic)

            self.IC.append(np.mean(ics))

        self.IC = np.array(self.IC)
        self.kept_idx = np.argsort(-self.IC)[:self.k]
        
        if self.verbose == True and self.feature_names is not None:
            print('    the following features selected')
            kept_features = [self.feature_names[idx] for idx in self.kept_idx]
            ICs = [self.IC[idx] for idx in self.kept_idx]
            for feature, ic in zip(kept_features, ICs):
                print('    feature={:s}, IC={:.5f}'.format(feature, ic))

        return self
    
        

    def transform(self, X, y=None):
        
        
        return X[:, self.kept_idx]

class SelectKBestWeightedR2(TransformerMixin, BaseEstimator):
    '''
    Select k features with the best insample weighted r2_score

    Parameters
    ---------------
    k: int
        select best k features, if k=-1 donot select
        
    sample_weight: array-like
        the weight to compute r2

    feature_names: array-like
        list of feature names if you want to know which feature you've selected

    verbose: bool, default=False
        print the result of selection
    '''
    
    def __init__(self, k, sample_weight=None, feature_names=None, verbose=False):

        self.k = k
        self.sample_weight = sample_weight
        self.feature_names = feature_names
        self.verbose = verbose
    
    def fit(self,X,y):
        
        n_features = X.shape[1]
        
        if self.k >= X.shape[1] or self.k == -1:
            # selected number exceed the feature number
            # do nothing
            self.kept_idx = range(X.shape[1])
            return self
        
        self.kept_idx = []
        # store the r2_scores
        r2_scores = []
        
        for i in range(n_features):
            
            lr = LinearRegression()
            lr.fit(X[:,i].reshape((-1,1)), y, sample_weight=self.sample_weight)
            y_pred = lr.predict(X[:,i].reshape((-1,1)))
            r2 = r2_score(y_true=y, y_pred=y_pred, sample_weight=self.sample_weight)
            r2_scores.append(r2)
            
        self.r2_scores = np.array(r2_scores)
            
        # np.argsort sort the array in ascending order
        # so we negate the array so that the lowest 
        # elements become the highest elements and vice-versa 
        sorted_idx = np.argsort(-self.r2_scores)[:self.k]
        
        for idx in sorted_idx:
            self.kept_idx.append(idx)
            
        if self.verbose == True and self.feature_names is not None:
            print('    the following features selected')
            kept_features = [self.feature_names[idx] for idx in self.kept_idx]
            r2s = [self.r2_scores[idx] for idx in self.kept_idx]
            for feature, r2 in zip(kept_features, r2s):
                print('    feature={:s}, weighted r2={:.7f}'.format(feature, r2))
            
        return self
            
    def transform(self, X, y=None):
        
        return X[:,self.kept_idx]

class GridSearchWeightedCV(BaseEstimator):

    '''
    GridSearch with cross-validation and sample weight

    Parameters:
    ------------------
    estimator: TransformedTargetRegressor
        the estimator to search, must implement fit(), predict() and score()
        the regressor of this TransformedTargetRegressor must be a pipeline

    param_grid: dict
        the grid of parameters to search on

    regressor_name: str
        the name of the regressor (final step of the pipeline)

    refit: bool, default=True
        if True, refit the best parameters on the entire training set

    random_state: int
        random_state for the shuffle

    cv: int, default=5
        number of folds for the k-fold cross-validation

    n_jobs: int, default=1
        number of process to run, not yet implemented

    shuffle: bool, default=True
        if True, shuffle the date

    eval_set: bool, default=False
        if True, pass the validation set to the eval_set fit_params (for xgboost early stopping)

    '''

    def __init__(self, estimator, param_grid, refit=True, 
                 random_state=None, cv=5, n_jobs=1, shuffle=True, 
                 eval_set=False, verbose=False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.eval_set = eval_set
        self.random_state = random_state
        self.refit = refit
        self.verbose = verbose
        # if the input estimator contains a SelectKBestIC transformer
        # since we need to pass some arguments to this 
        self.contains_ic_selector = False
        self.ic_selector = None
        self.contains_r2_selector = False
        self.r2_selector = None
        for name, transformer in estimator.regressor.steps:
            if isinstance(transformer, SelectKBestIC):
                self.contains_ic_selector = True
                self.ic_selector = name
            if isinstance(transformer, SelectKBestWeightedR2):
                self.contains_r2_selector = True
                self.r2_selector = name
                
        # get the regressor's name from the pipeline
        self.regressor_name = estimator.regressor.steps[-1][0]
        
    def _refit(self, X, y, Date, Id, sample_weight, fit_params):
        
        '''
        refit a model
        '''
        
        if sample_weight is not None:
            prefix = self.regressor_name+'__' if self.eval_set==False else ''
            fit_params[prefix + 'sample_weight'] = sample_weight
        
        # no evaluation set, just fit the pipeline
        if self.eval_set == False:
            
            # setup the selector's parameter
            params = {}
            for old_key in self.best_params_.keys():
                params['regressor__'+old_key] = self.best_params_[old_key]
                
            if self.contains_ic_selector:
                params['regressor__{:s}__Id'.format(self.ic_selector)] = Id
                params['regressor__{:s}__Date'.format(self.ic_selector)] = Date
                params['regressor__{:s}__feature_names'.format(self.ic_selector)] = list(X)
            
            self.estimator.set_params(**params)
            self.best_estimator = self.estimator.fit(
                X, y, **fit_params)
            
            return
        
        # with evaluation set, breakup the pipeline
        preprocessor = self.estimator.regressor[:-1]
        regressor = self.estimator.regressor[-1]
        
        # clip the target
        y_train_processed = self.estimator.func(y)
        y_valid_processed = y_train_processed
        # transform X
        X_train_processed = preprocessor.fit_transform(X, y_train_processed)
        X_valid_processed = X_train_processed
        # setup the eval_set as the transformed X and y on valid set
        fit_params['eval_set'] = [[X_valid_processed, y_valid_processed]]
         
        # setup the parameters
        # notice that we need to remove all the name prefix
        params = {}
        for key, value in self.best_params_.items():
            key = key.replace(self.regressor_name+'__', '')
            params[key] = value
        regressor.set_params(**params)
         # directly fit the regressor
        regressor.fit(X_train_processed, y_train_processed, **fit_params)
        self.best_estimator = FittedTransformedTargetRegressor(
            preprocessor=preprocessor,
            regressor=regressor,
            func=self.estimator.func
        )
        
        return
        

    def fit(self, X, y, Date, Id, sample_weight=None, **fit_params):
        '''

        Parameters
        ----------------------
        X: pd.Dataframe 
            training set

        y: pd.Dataframe
            target set

        Date: pd.Dataframe
            The date series for each row in X

        Id: pd.Dataframe
            the id series for each row in X

        sample_weight: array-like
            the weight for training, must in the same shape of y

        fit_params: dict
            the fit_params for the regressor (final step of the pipeline),
            please refer to the documentation of the regressor of your choice

        '''

        cv_score_list = []
        old_fit_params = fit_params.copy()
        
        for old_params in ParameterGrid(self.param_grid):
            # add a regressor behind the key
            # required by set_params() method
            if self.verbose:
                print(old_params)
            params = {}
            for old_key in old_params.keys():
                params['regressor__'+old_key] = old_params[old_key]
            cv_score_sum = 0
            # cross-validation on dates
            dates = Date.unique()
            kf = KFold(n_splits=self.cv, shuffle=self.shuffle,
                       random_state=self.random_state)
            
            
            for k, (train_date_index, valid_date_index) in enumerate(kf.split(dates)):
                        
                train_index = Date.isin(dates[train_date_index])
                valid_index = Date.isin(dates[valid_date_index])

                X_train = X.loc[train_index]
                X_valid = X.loc[valid_index]

                y_train = y.loc[train_index]
                y_valid = y.loc[valid_index]

                Id_train = Id.loc[train_index]
                Date_train = Date.loc[train_index]

                # setup the IC selector's parameter if need
                if self.contains_ic_selector:
                    params['regressor__{:s}__Id'.format(self.ic_selector)] = Id_train
                    params['regressor__{:s}__Date'.format(self.ic_selector)] = Date_train
                    params['regressor__{:s}__feature_names'.format(self.ic_selector)] = list(X_train)
                
                
                
                # if contains sample_weight: True
                # then add our weight
                if sample_weight is not None:
                    w_train, w_valid = sample_weight.loc[train_index], sample_weight.loc[valid_index]
                    
                    
                    # this if-else is important
                    # if we use eval_set on xgboost, we have to break-out the pipeline
                    # since we cannot add the pipeline in the evaluation process
                    # so the fit parameters shall not contains regressor name
                    # if we use xgboost with eval_set
                    
                    if self.eval_set == False:
                        # add the regressor's name to the arguments
                        fit_params = {}
                        for old_key in old_fit_params.keys():
                            fit_params[self.regressor_name+'__'+old_key] = old_fit_params[old_key]
                        fit_params[self.regressor_name+'__sample_weight'] = w_train
                    else:
                        fit_params['sample_weight'] = w_train
                else:
                    w_train = None
                    w_valid = None
                    
                # setup the R2 selector's parameter if need
                if self.contains_r2_selector:
                    params['regressor__{:s}__sample_weight'.format(self.r2_selector)] = w_train
                    params['regressor__{:s}__feature_names'.format(self.r2_selector)] = list(X_train)
                
                # clone an estimator to fit for multiple times
                estimator = clone(self.estimator)
                estimator.set_params(**params)

                # if we choose an eval_set
                # breakup the pipeline into preprocessing and regressor
                # add the eval_set into the last step of the estimator
                if self.eval_set == True:
                    preprocessor = estimator.regressor[:-1]
                    regressor = estimator.regressor[-1]
                    # clip the target
                    y_train_processed = estimator.func(y_train)
                    y_valid_processed = estimator.func(y_valid)
                    # transform X
                    X_train_processed = preprocessor.fit_transform(X_train, y_train_processed)
                    X_valid_processed = preprocessor.transform(X_valid)
                    # setup the eval_set as the transformed X and y on valid set
                    fit_params['eval_set'] = [[X_valid_processed, y_valid_processed]]
                    # directly fit the regressor
                    regressor.fit(X_train_processed, y_train_processed, **fit_params)
                    cv_score = regressor.score(X_valid_processed, 
                                               y_valid_processed, 
                                               sample_weight=w_valid)
                    train_score = regressor.score(X_train_processed, 
                                                  y_train_processed, 
                                                  sample_weight=w_train)
                else:
                    # if no eval set, then fit the whole pipeline
                    estimator.fit(X_train, y_train, **fit_params)
                    # use the weight provided to compute the r2_score
                    cv_score = estimator.score(X_valid, y_valid, sample_weight=w_valid)
                    train_score = estimator.score(X_train, y_train, sample_weight=w_train)
                
                if self.verbose:    
                    print('    fold={:d}, train score={:.7f}, cv score={:.7f}\n'.format(k, train_score, cv_score))
                
                cv_score_sum += cv_score

            cv_score_sum /= self.cv
            cv_score_list.append(cv_score_sum)

        self.best_score_ = np.max(cv_score_list)
        self.best_params_ = ParameterGrid(self.param_grid)[np.argmax(cv_score_list)]
        
        # refit the model based on the best parameter on the whole training set
        if self.refit == True:
            if self.verbose:
                print('cross-validation completed, refitting on the whole train set')
                print('best parameter:')
                print(self.best_params_)
            self._refit(X, y, Date, Id, sample_weight, old_fit_params)

        return self