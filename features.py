from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from tqdm import tqdm




class FeatureExtractor(ABC):

    '''
    extract the features without any leakage
    '''

    def __init__(self, feature_name, input_dir, output_dir):
        self.feature_name = feature_name
        self.input_dir = input_dir
        self.output_dir = output_dir

    def feed(self, data):
        self.data = data
        self.date_index_list = data['MDV_63'].index.to_list()
        return self

    @abstractmethod
    def transform(self, date):
        '''
        calculate the feature at 16:00 on the date specified
        using the data available prior to date (inclusive)
        return a dataframe in the shape of

            feature_name
        id1
        id2
        ...
        idk


        REMEMBER NO DATA LEAKAGE (both time series and cross-sectional)
        feature at t can only be constructed from data avaliable before t1

        if you need more data than the data available (for example you want
        to compute moving average on 20140102 (first date)), you should just
        return a dataframe with NaN
        '''
        pass

    def extract(self, start_date, end_date, save=True, disable_progress=False):

        dates = [date for date in self.data['CleanMid_open'].index if (date >= start_date) and (date <= end_date)]
        loop = tqdm(dates, disable=disable_progress, leave=False)
        for date in loop:
            self.transform(date, save)
            loop.set_description('extracting {:s}'.format(self.feature_name))

    def save(self, df_feature, date):
        '''
        save the feature to the output dir

        Parameters
        ----------
        df_feature : pd.DataFrame
            dataframe of the feature, index must be Id, 
            e.g.:

                          feature_name
            Id                       
            ID000BC0991   1201.894492
            ID000BC0KZ7   4948.177745
            ID000BC0QR3    705.254172
            ID000BC1099    722.692678
            ID000BC12C1   1849.788102

        date: str
            the date of the feature (since we're supposed to save a csv for each date)

        '''


        filename = self.output_dir+'/'+date+'.csv'

        if os.path.isfile(filename) == False:
            # create the new csv file if not exist
            # align the prediction with residual
            ids_on_date = self.data['ResidualNoWinsorCumReturn_close'].loc[date].dropna().index.to_frame(name='ret')
            # use left join since we only predict the Ids of residual
            feature = ids_on_date.join(df_feature, how='left')
            # remove the residual column
            feature.drop(columns='ret', inplace=True)
            feature.to_csv(filename)
        else:
            # if exist, modify it
            feature = pd.read_csv(filename).set_index('Id')
            if self.feature_name in list(feature):
                # remove the old version feature
                feature.drop(columns=self.feature_name, inplace=True)
            feature = feature.join(df_feature, how='left')
            feature.to_csv(filename)

    
    
    
class YesterdayRet(FeatureExtractor):
    
    '''
    Actual return from previous close to the time of prediction 
    (i.e. at 16:00), to make the drift plot
    '''
    
    def transform(self, date, save=True):
        
        ret = self.data['ResidualNoWinsorCumReturn_close'].loc[date]
        df_feature = ret.to_frame(name=self.feature_name)
        if save == True:
            self.save(df_feature, date)
        return df_feature
    
    
    
        
class TodaytoNextMorningRet(FeatureExtractor):
    
    '''
    return from ”today’s” close to the next morning at 10:00
    '''
    
    def transform(self, date, save=True):
        date_idx = list(self.data['ResidualNoWinsorCumReturn_close'].index).index(date)
        ret_open = self.data['ResidualNoWinsorCumReturn_open'].iloc[date_idx+1]
        df_feature = ret_open.to_frame(name=self.feature_name)
        if save == True:
            self.save(df_feature, date)
        return df_feature
    
    
    
    
class NextIntradayRet(FeatureExtractor):
    
    '''
    return from 10:00 16:00 on the next day
    '''
    
    def transform(self, date, save=True):
        date_idx = list(self.data['ResidualNoWinsorCumReturn_close'].index).index(date)
        ret_open = self.data['ResidualNoWinsorCumReturn_open'].iloc[date_idx+1]
        ret_close = self.data['ResidualNoWinsorCumReturn_close'].iloc[date_idx+1]
        ret = ret_close - ret_open
        df_feature = ret.to_frame(name=self.feature_name)
        if save == True:
            self.save(df_feature, date)
        return df_feature



    
class InversedVol(FeatureExtractor):
    
    '''
    1/estVol
    '''
    
    def transform(self, date, save=True):
        
        estvol = self.data['estVol'].loc[date]
        inversed_estvol = 1 / estvol
        inversed_estvol.replace(np.inf, 0, inplace=True)
        inversed_estvol.replace(-np.inf, 0, inplace=True)
        inversed_estvol.replace(np.nan, 0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = inversed_estvol.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature
    
    
    
    
class corr_X_Y(FeatureExtractor):

    '''corr(X, Y, N)'''

    def __init__(self, feature_name, input_dir, output_dir, X_name, Y_name, N, X_rank=False, Y_rank=False):
        super(corr_X_Y, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.Y_name = Y_name
        self.N = N
        self.X_rank = X_rank
        self.Y_rank = Y_rank

    def transform(self, date, save=True):

        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.X_rank == True:
            X = X.rank(axis='columns')

        # Y
        Y = self.data[self.Y_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.Y_rank == True:
            Y = Y.rank(axis='columns')

        # corr(X, Y, N)
        corr_X_Y = ((X * Y).mean() - X.mean() * Y.mean()) / X.std() / Y.std()
        corr_X_Y.replace(np.inf, np.nan, inplace=True)
        corr_X_Y.replace(-np.inf, np.nan, inplace=True)
        corr_X_Y[corr_X_Y > 1] = 1
        corr_X_Y[corr_X_Y < -1] = -1
#         corr_X_Y.fillna(method='ffill', inplace=True)
#         corr_X_Y.fillna(0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = corr_X_Y.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature


    
    
class ts_sum_X(FeatureExtractor):

    '''ts_sum(X, N)'''

    def __init__(self, feature_name, input_dir, output_dir, X_name, N, X_rank=False):
        super(ts_sum_X, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.N = N
        self.X_rank = X_rank

    def transform(self, date, save=True):

        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.X_rank == True:
            X = X.rank(axis='columns')

        # ts_sum(X, N)
        ts_sum_X = X.sum()
        ts_sum_X.replace(np.inf, np.nan, inplace=True)
        ts_sum_X.replace(-np.inf, np.nan, inplace=True)
#         ts_sum_X.fillna(method='ffill', inplace=True)
#         ts_sum_X.fillna(0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = ts_sum_X.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature

    
    
    
class ts_sum_X_Y(FeatureExtractor):

    '''
        ts_sum(X_and_Y, N)
            if and = '+', return X+Y
            if and = '-', return X-Y
            if and = '*', return X*Y
            if and = '/', return X/Y
    '''

    def __init__(self, feature_name, input_dir, output_dir, X_name, Y_name, N, __and__=np.multiply, X_rank=False, Y_rank=False):
        super(ts_sum_X_Y, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.Y_name = Y_name
        self.N = N
        self.__and__ = __and__
        self.X_rank = X_rank
        self.Y_rank = Y_rank

    def transform(self, date, save=True):

        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.X_rank == True:
            X = X.rank(axis='columns')

        # Y
        Y = self.data[self.Y_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.Y_rank == True:
            Y = Y.rank(axis='columns')

        # ts_sum(X * Y, N)
        ts_sum_X_Y = self.__and__(X, Y).sum()
        ts_sum_X_Y.replace(np.inf, np.nan, inplace=True)
        ts_sum_X_Y.replace(-np.inf, np.nan, inplace=True)
#         ts_sum_X_Y.fillna(method='ffill', inplace=True)
#         ts_sum_X_Y.fillna(0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = ts_sum_X_Y.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature

    
    
    
class X_weighted_Y(FeatureExtractor):

    '''ts_sum(X * Y, N) / ts_sum(X, N)'''

    def __init__(self, feature_name, input_dir, output_dir, X_name, Y_name, N, X_rank=False, Y_rank=False):
        super(X_weighted_Y, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.Y_name = Y_name
        self.N = N
        self.X_rank = X_rank
        self.Y_rank = Y_rank

    def transform(self, date, save=True):

        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.X_rank == True:
            X = X.rank(axis='columns')

        # Y
        Y = self.data[self.Y_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.Y_rank == True:
            Y = Y.rank(axis='columns')

        # ts_sum(X * Y, N) / ts_sum(X, N)
        X_weighted_Y = (X * Y).sum() / X.sum()
        X_weighted_Y.replace(np.inf, np.nan, inplace=True)
        X_weighted_Y.replace(-np.inf, np.nan, inplace=True)
#         X_weighted_Y.fillna(method='ffill', inplace=True)
#         X_weighted_Y.fillna(0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = X_weighted_Y.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature
    
    
    
    
class X_and_Y(FeatureExtractor):

    '''
        X_and_Y
            if and = '+', return X+Y
            if and = '+', return X+Y
            if and = '+', return X+Y
            if and = '+', return X+Y
    '''

    def __init__(self, feature_name, input_dir, output_dir, X_name, Y_name, __and__, X_rank=False, Y_rank=False):
        super(X_and_Y, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.Y_name = Y_name
        self.__and__ = __and__
        self.X_rank = X_rank
        self.Y_rank = Y_rank

    def transform(self, date, save=True):

        # X
        X = self.data[self.X_name].loc[date]
        if self.X_rank == True:
            X = X.rank(axis='columns')

        # Y
        Y = self.data[self.Y_name].loc[date] 
        if self.Y_rank == True:
            Y = Y.rank(axis='columns')

        # X_and_Y
        X_and_Y = self.__and__(X, Y)
        X_and_Y.replace(np.inf, np.nan, inplace=True)
        X_and_Y.replace(-np.inf, np.nan, inplace=True)
#         X_and_Y.fillna(method='ffill', inplace=True)
#         X_and_Y.fillna(0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = X_and_Y.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature

    
    
    
class Alligator(FeatureExtractor):
    
    '''
        Williams Alligator Indicator
        
        Alligator's Jaw / Blue Line:
            13-period Smoothed Moving Average, moved by 8 bars into the future
        Alligator's Teeth / Red Line:
            8-period Smoothed Moving Average, moved by 5 bars into the future
        Alligator's Lips / Green Line:
            5-period Smoothed Moving Average, moved by 3 bars into the future
    '''

    def __init__(self, feature_name, input_dir, output_dir, X_name='CleanMid_close', color='lips-teeth'):
        super(Alligator, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.color = color

    def transform(self, date, save=True):

        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index-12: date_index+1, :] 
        
        # Williams Alligator Indicator
        if self.color == 'jaw':
            Alligator = X.iloc[-13:,:].mean()
        elif self.color == 'teeth':
            Alligator = X.iloc[-8:,:].mean()
        elif self.color == 'lips':
            Alligator = X.iloc[-5:,:].mean()
        elif self.color == 'lips-teeth':
            Alligator = X.iloc[-5:,:].mean() - X.iloc[-8:,:].mean()
        elif self.color == 'lips-jaw':
            Alligator = X.iloc[-5:,:].mean() - X.iloc[-13:,:].mean()
        else:
            print('No such color!')
            return
        Alligator /= X.mean()
        Alligator.replace(np.inf, np.nan, inplace=True)
        Alligator.replace(-np.inf, np.nan, inplace=True)
#         Alligator.fillna(method='ffill', inplace=True)
#         Alligator.fillna(0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = Alligator.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature
    
    
    
    
class Accelerator_Oscillator(FeatureExtractor):

    '''
        Accelerator Oscillator (AC)
        Awesome Oscillator (AO)
        
            MEDIAN PRICE = (HIGH + LOW) / 2
            AO = SMA (MEDIAN PRICE, 5) - SMA (MEDIAN PRICE, 34)
            AC = AO - SMA (AO, 5)
    '''

    def __init__(self, feature_name, input_dir, output_dir, X_name='CleanMid_close'):
        super(Accelerator_Oscillator, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        
    def transform(self, date, save=True):

        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index-38: date_index+1, :] 
        
        # Accelerator Oscillator (AC)
        AO_0 = X.iloc[-5:, :].mean() - X.iloc[-34:,:].mean()
        AO_1 = X.iloc[-6:-1, :].mean() - X.iloc[-35:-1,:].mean()
        AO_2 = X.iloc[-7:-2, :].mean() - X.iloc[-36:-2,:].mean()
        AO_3 = X.iloc[-8:-3, :].mean() - X.iloc[-37:-3,:].mean()
        AO_4 = X.iloc[-9:-4, :].mean() - X.iloc[-38:-4,:].mean()
        AC = AO_0 - (AO_0 + AO_1 + AO_2 + AO_3 + AO_4) / 5
        AC /= X.mean()
        AC.replace(np.inf, np.nan, inplace=True)
        AC.replace(-np.inf, np.nan, inplace=True)
#         AC.fillna(method='ffill', inplace=True)
#         AC.fillna(0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = AC.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature

    
    
        
class ts_skew_X(FeatureExtractor):

    '''
        ts_skew(X,N)
    '''

    def __init__(self, feature_name, input_dir, output_dir, X_name, N, X_rank=False, feature_rank=False):
        super(ts_skew_X, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.N = N
        self.X_rank = X_rank
        self.feature_rank = feature_rank
        
    def transform(self, date, save=True):

        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.X_rank == True:
            X = X.rank(axis='columns')
        
        # ts_skew(X,N)
        ts_skew_X = X.skew()
        ts_skew_X.replace(np.inf, np.nan, inplace=True)
        ts_skew_X.replace(-np.inf, np.nan, inplace=True)
#         ts_skew_X.fillna(method='ffill', inplace=True)
#         ts_skew_X.fillna(0, inplace=True)
        if self.feature_rank == True:
            ts_skew_X = ts_skew_X.rank()                       # feature rank
    
        # convert the pd.Series into pd.Dataframe
        df_feature = ts_skew_X.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature
    
    
    
    
class ts_kurt_X(FeatureExtractor):

    '''
        ts_kurt(X,N)
    '''

    def __init__(self, feature_name, input_dir, output_dir, X_name, N, X_rank=False, feature_rank=False):
        super(ts_kurt_X, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.N = N
        self.X_rank = X_rank
        self.feature_rank = feature_rank

    def transform(self, date, save=True):

        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.X_rank == True:
            X = X.rank(axis='columns')
        
        # ts_kurt(X,N)
        ts_kurt_X = X.kurt()
        ts_kurt_X.replace(np.inf, np.nan, inplace=True)
        ts_kurt_X.replace(-np.inf, np.nan, inplace=True)
#         ts_kurt_X.fillna(method='ffill', inplace=True)
#         ts_kurt_X.fillna(0, inplace=True)
        if self.feature_rank == True:
            ts_kurt_X = ts_kurt_X.rank()                       # feature rank

        # convert the pd.Series into pd.Dataframe
        df_feature = ts_kurt_X.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature
    
    
    
    
#################################################################################
#Kaylie 5.10
##################################Trend Factors##################################
class MACD(FeatureExtractor):
    '''
    Moving Average Convergence/Divergence, used to spot changes in the strength, 
    direction, momentum, and duration of a trend in a stock's price.
    
    The calculation of MACD contains three part:
    A = the difference between two exponential moving averages (EMAs) of closing prices. 
    B = exponential moving average of A. 
    MACD = the divergence between A and B, i.e. A-B
    '''
    def __init__(self, feature_name, input_variable_name, input_dir, output_dir, faster, slower, signal):
        super(MACD, self).__init__(feature_name, input_dir, output_dir)
        self.input_variable_name = input_variable_name
        self.faster = faster
        self.slower = slower
        self.signal = signal
    
    def feed(self, data):
        self.data = data
        self.A = pd.DataFrame(columns = self.data[self.input_variable_name].columns)
        return self
    
    def _EMA(self,date,window,dt):
        
        '''
        Exponential moving average of past n days.
        
        '''
        #the row number of date
        n = np.where(dt.index == date)[0][0]
    
        if n >= window:
            return dt.iloc[n-window+1:n+1,:].ewm(span = window).mean().iloc[-1,:]
        else:
            columns = dt.columns
            return pd.Series([None for i in range(len(columns))],index=columns)
    
    def transform(self, date, save=True):
        
        dt = self.data[self.input_variable_name]
        self.A.loc[date] = self._EMA(date,self.faster,dt)-self._EMA(date,self.slower,dt)
        B = self._EMA(date,self.signal,self.A)
        MACD = self.A.loc[date] - B 
        
        df_feature = MACD.to_frame(name=self.feature_name)
        
        if save == True:            
            self.save(df_feature, date)
            
        return df_feature     
    

    
    
class Reversal(FeatureExtractor):
    '''
    Get the rank of a stock in terms of return, then reverse the rank and get its percentile.
    '''
    def __init__(self, feature_name, input_dir, output_dir, window):
        super(Reversal, self).__init__(feature_name, input_dir, output_dir)
        self.window = window
    
    def transform(self, date, save=True):
        
        dt = self.data['ResidualNoWinsorCumReturn_close']
        n = np.where(dt.index == date)[0][0]
        
        if n >= self.window:
            ret = dt.iloc[n-self.window]
            rev = ret.rank(pct=True,ascending=False)
        else:
            rev = pd.Series([None for i in range(len(dt.columns))],index=dt.columns)
            
        df_feature = rev.to_frame(name=self.feature_name)
        
        if save == True:            
            self.save(df_feature, date)
            
        return df_feature
    
    

    
class EMA(FeatureExtractor):
    '''
    Exponential Moving Average
    '''
    def __init__(self, feature_name, input_variable_name, input_dir, output_dir, window):
        super(EMA, self).__init__(feature_name, input_dir, output_dir)
        self.input_variable_name = input_variable_name
        self.window = window
    
    def _EMA(self,date,window,dt):
        
        '''
        Exponential moving average of past n days.
        
        '''
        #the row number of date
        n = np.where(dt.index == date)[0][0]
    
        if n >= window:
            return dt.iloc[n-window+1:n+1,:].ewm(span = window).mean().iloc[-1,:]
        else:
            columns = dt.columns
            return pd.Series([None for i in range(len(columns))],index=columns)
    
    def transform(self, date, save=True):
        
        dt = self.data[self.input_variable_name]
        ema = self._EMA(date,self.window,dt)
        
        df_feature = ema.to_frame(name=self.feature_name)
        
        if save == True:            
            self.save(df_feature, date)
            
        return df_feature     
    
    
    

class MQ(FeatureExtractor):
    
    '''
        moving quantile
        MQ(N) = Quantile(  p_{t},p_{t-1},...p{t-N+1}  , first_quantile) - Quantile(  p_{t},p_{t-1},...p{t-N+1}  , second_quantile)
    '''
    
    def __init__(self, feature_name, input_dir, output_dir, X_name, N, X_rank=False, first_quantile=0.5, second_quantile=None):
        super(MQ, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.N = N
        self.X_rank = X_rank
        self.first_quantile = first_quantile      
        self.second_quantile = second_quantile      
    
    def transform(self, date, save=True):
        
        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.X_rank == True:
            X = X.rank(axis='columns')
        
        # moving quantile
        MQ = X.quantile(self.first_quantile)
        if self.second_quantile is not None:
            MQ -= X.quantile(self.second_quantile)
        MQ.replace(np.inf, np.nan, inplace=True)
        MQ.replace(-np.inf, np.nan, inplace=True)
#         MQ.fillna(method='ffill', inplace=True)
#         MQ.fillna(0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = MQ.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature
    
    
    
    
class DataNdaysBefore(FeatureExtractor):

    def __init__(self, feature_name, input_variable_name, input_dir, output_dir, window):
        super(DataNdaysBefore, self).__init__(feature_name, input_dir, output_dir)
        self.window = window
        self.input_variable_name = input_variable_name
     
    def transform(self, date, save=True):
        
        dt = self.data[self.input_variable_name]
        n = np.where(dt.index == date)[0][0]
        
        if n >= self.window:
            feature = dt.iloc[n-self.window,:]
        else:
            columns = dt.columns
            feature = pd.Series([None for i in range(len(columns))],index=columns)
        
        # convert the pd.Series into pd.Dataframe
        df_feature = feature.to_frame(name=self.feature_name)
        
        if save == True:            
            self.save(df_feature, date)
            
        return df_feature

    
    
    
class TrendNdays(FeatureExtractor):

    def __init__(self, feature_name, input_variable_name, input_dir, output_dir, window):
        super(TrendNdays, self).__init__(feature_name, input_dir, output_dir)
        self.window = window
        self.input_variable_name = input_variable_name
     
    def transform(self, date, save=True):
        
        dt = self.data[self.input_variable_name]
        n = np.where(dt.index == date)[0][0]
        
        if n >= self.window:
            feature = dt.iloc[n-self.window:n,:].sum()
        else:
            columns = dt.columns
            feature = pd.Series([None for i in range(len(columns))],index=columns)
        
        # convert the pd.Series into pd.Dataframe
        df_feature = feature.to_frame(name=self.feature_name)
        
        if save == True:            
            self.save(df_feature, date)
            
        return df_feature
    
    


##################################Volume Factors##################################    
class OBV(FeatureExtractor):
    
    '''
    On-balance volume (OBV) is a technical analysis indicator intended to 
    relate price and volume in the stock market.
    OBV is based on a cumulative total volume.
    
                     volumn, if close>close_prev
    OBV = OBV_prev + 0,      if close=close_prev
                     -volumn,if close<close_prev
    '''
    
    def __init__(self, feature_name, input_dir, output_dir, X_name='CumVolume_close', Y_name='RawNoWinsorCumReturn_close', N=20, X_rank=False, Y_rank=False):
        super(OBV, self).__init__(feature_name, input_dir, output_dir)
        self.X_name = X_name
        self.Y_name = Y_name
        self.N = N
        self.X_rank = X_rank
        self.Y_rank = Y_rank
   
    def transform(self, date, save=True):
        
        # index number of current date
        date_index = self.date_index_list.index(date)
        
        # X
        X = self.data[self.X_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.X_rank == True:
            X = X.rank(axis='columns')

        # Y
        Y = self.data[self.Y_name].iloc[date_index+1 - self.N: date_index+1, :] 
        if self.Y_rank == True:
            Y = Y.rank(axis='columns')
            
        # OBV
        OBV = (X * np.sign(Y)).sum()
        OBV.replace(np.inf, np.nan, inplace=True)
        OBV.replace(-np.inf, np.nan, inplace=True)
#         OBV.fillna(method='ffill', inplace=True)
#         OBV.fillna(0, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = OBV.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)

        return df_feature




class Liquidity(FeatureExtractor):
    '''
    Liquidity is used to value the cash flow liquidity of a single stock, 
    used to see whether it has large trading volume on average on the past several days.
    We define it as ln(V_t + V_t-1 + ... + V_t-n+1), where V_t is trading volume in shares.
    '''
    def __init__(self, feature_name, input_variable_name, input_dir, output_dir, window):
        super(Liquidity, self).__init__(feature_name, input_dir, output_dir)
        self.window = window
        self.input_variable_name = input_variable_name

    def _calculate_liquidity(self,date,window,dt):
        
        #the row number of date
        n = np.where(dt.index == date)[0][0]
    
        if n >= window:
            liq = dt.iloc[n-window+1:n+1,:].sum(axis=0)
            log_liq = liq.apply(np.log)
            log_liq.replace(np.inf, np.nan, inplace=True)
            log_liq.replace(-np.inf, np.nan, inplace=True)
            return log_liq
        else:
            columns = dt.columns
            return pd.Series([None for i in range(len(columns))],index=columns)
       
    def transform(self, date, save=True):
        
        MM = self._calculate_liquidity(date,self.window,self.data[self.input_variable_name])
        
        # convert the pd.Series into pd.Dataframe
        df_feature = MM.to_frame(name=self.feature_name)
        
        if save == True:            
            self.save(df_feature, date)
            
        return df_feature

    
    
    
##################################Volatility Factors##################################
class BB(FeatureExtractor):
    '''
    Bollinger Bands. It can be used to measure the highness or lowness of the 
    price relative to previous trades.
    Bollinger Bands consist of:
    - a middle band being an N-period simple moving average (MA)
    - an upper band at K times an N-period standard deviation above the middle band (MA + Kσ)
    - a lower band at K times an N-period standard deviation below the middle band (MA − Kσ)
    Typical values for N and K are 20 and 2, respectively.
    
    %b = (last − lowerBB) / (upperBB − lowerBB)
    '''
    def __init__(self, feature_name, input_variable_name, input_dir, output_dir, window, k):
        super(BB, self).__init__(feature_name, input_dir, output_dir)
        self.window = window
        self.input_variable_name = input_variable_name
        self.k = k
    
    def _EMA(self,date,window,dt):
        
        '''
        Exponential moving average of past n days.
        '''
        #the row number of date
        n = np.where(dt.index == date)[0][0]
    
        if n >= window:
            return dt.iloc[n+1-window:n+1,:].ewm(span = window).mean().iloc[-1,:]
        else:
            columns = dt.columns
            return pd.Series([None for i in range(len(columns))],index=columns)
 
    def transform(self, date, save=True):
        
        sigma = self.data['estVol'].loc[date]
        ema = self._EMA(date,self.window,self.data[self.input_variable_name])
        last = self.data[self.input_variable_name].loc[date]
        lowerBB = ema-self.k*sigma 
        upperBB = ema+self.k*sigma 
        b_percent = (last-lowerBB)/(upperBB-lowerBB)
        
        # convert the pd.Series into pd.Dataframe
        df_feature = b_percent.to_frame(name=self.feature_name)
        
        if save == True:            
            self.save(df_feature, date)
            
        return df_feature
  



###############################Alpha 101 Series#############################
class Alpha015(FeatureExtractor):
    '''
    rank(correlation(rank(close), rank(volume), n))
    '''
    def __init__(self, feature_name, input_dir, output_dir, window):
        super(Alpha015, self).__init__(feature_name, input_dir, output_dir)
        self.window = window  
        
    def feed(self, data):
        self.data = data
        self.date_index_list = data['MDV_63'].index.to_list()
        self.rank_midclose = pd.DataFrame(columns = self.data['CleanMid_close'].columns)
        self.rank_volume = pd.DataFrame(columns = self.data['CumVolume_close'].columns)
        return self
    
    def transform(self, date, save=True):

        #the row number of date
        
        close = self.data['CleanMid_close']
        volume = self.data['CumVolume_close']
        
        n = np.where(close.index == date)[0][0]
        self.rank_midclose.loc[date] = close.loc[date].rank(pct=True)
        self.rank_volume.loc[date] = volume.loc[date].rank(pct=True)
 
        if n < self.window:
            correlation = pd.Series([None for i in range(len(close.columns))],index = close.columns) 
        else:
            correlation = pd.Series([self.rank_midclose.iloc[-self.window:,i].corr(self.rank_volume.iloc[-self.window:,i]) 
                                        for i in range(len(self.rank_midclose.columns))],index = close.columns).rank(pct=True)
            
        # convert the pd.Series into pd.Dataframe
        df_feature = correlation.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)
        
        return df_feature 
    
    
    

#################################################################################
# Kaylie 2021/5/16
#################################################################################
class Squared(FeatureExtractor):
    '''
    log(-log(ret**2))
    '''
    def __init__(self, feature_name, input_dir, output_dir, input_variable):
        super(Squared, self).__init__(feature_name, input_dir, output_dir)
        self.input_variable = input_variable

    def transform(self, date, save=True):

        cleanmid = self.data[self.input_variable].loc[date]
        squared_cm = np.log(-np.log(cleanmid ** 2))
        squared_cm.replace(np.inf, np.nan, inplace=True)
        squared_cm.replace(-np.inf, np.nan, inplace=True)

        # convert the pd.Series into pd.Dataframe
        df_feature = squared_cm.to_frame(name=self.feature_name)

        if save == True:
            self.save(df_feature, date)

        return df_feature
    

    
    
class FD(FeatureExtractor):
    '''
    Fractional Differentiation
    '''
    def __init__(self, feature_name, input_dir, output_dir, input_variable, window, d):
        super(FD, self).__init__(feature_name, input_dir, output_dir)
        self.input_variable = input_variable
        self.window = window  
        self.d = d
    
    def transform(self, date, save=True):

        dt = self.data[self.input_variable]
        
        weights = [[1]]
        for k in range(1,self.window):
            weights.append([-weights[-1][0]*(self.d-k+1)/k])
        weights.reverse()
        
        #the row number of date
        n = np.where(dt.index == date)[0][0]
        
        if n < self.window:
            fd = pd.Series([None for i in range(len(dt.columns))],index = dt.columns) 
        else:
            fd = np.sum(dt.iloc[n-self.window+1:n+1,:]*np.array(weights))
            
        # convert the pd.Series into pd.Dataframe
        df_feature = fd.to_frame(name=self.feature_name)

        if save == True:            
            self.save(df_feature, date)
        
        return df_feature 
    
    
    
    