from os import listdir
import pandas as pd
from tqdm import tqdm
import os

class DataLoader:
    
    '''
    
    load the raw data from the directory provided
    
    return a list of tuple (name, pd.DataFrame), each dataframe is as follows

        id1  id2  id3  ...
        
    t1
    t2
    t3
    .
    .
    .

    
    '''
    
    def __init__(self, filepath):
        
        '''
        filepath: the directory of the raw dataset containing the five folders
        '''
        
        self.filepath = filepath
        self.features = ['mdv', 'price_volume', 'return', 'risk', 'shout']
        self.dates = [filename[:-4] for filename in sorted(listdir(filepath+'/mdv')) if len(filename) == 12]
        
                
        
    def _load_a_single_feature(self, start_date='20150102', end_date='20150110', folder_name='mdv', feature_name='MDV_63', how='inner', disable_progress=False):
        
        '''
        internal helper method, you should not call this function outside the class
        '''
        
        df_ret = None
        
        dates = [date for date in self.dates if (date >= start_date) and (date <= end_date)]
        
        old_feature_name = feature_name
        loop = tqdm(dates, leave=False)
        for date in loop:
            
            filename = self.filepath+'/'+folder_name+'/'+date+'.csv'
            if df_ret is None:
                df_ret = pd.read_csv(filename)
                # choose the time based on close or open keyworld
                if old_feature_name[-5:] == 'close':
                    df_ret = df_ret.loc[df_ret['Time'] == '16:00:00.000']
                    feature_name = old_feature_name[:-6]
                elif old_feature_name[-4:] == 'open':
                    df_ret = df_ret.loc[df_ret['Time'] == '10:00:00.000']
                    feature_name = old_feature_name[:-5]
                df_ret = df_ret[['Id', feature_name]]
                df_ret.set_index('Id', inplace=True)

                # change the feature name to date
                # since we're going to transpose the dataframe
                df_ret.rename(columns={feature_name:date}, inplace=True)
            else:
                
                df = pd.read_csv(filename)
                
                if old_feature_name[-5:] == 'close':
                    df = df.loc[df['Time'] == '16:00:00.000']
                    feature_name = old_feature_name[:-6]
                elif old_feature_name[-4:] == 'open':
                    df = df.loc[df['Time'] == '10:00:00.000']
                    feature_name = old_feature_name[:-5]
                
                df = df[['Id', feature_name]]
                df.rename(columns={feature_name:date}, inplace=True)
                df.set_index('Id', inplace=True)
                df_ret = df_ret.join(df, how=how)
                
            loop.set_description('Loading {:s}'.format(old_feature_name))
        # transpose the dataframe to satisfy the team members
        df_ret = df_ret.T
        
        return old_feature_name, df_ret
            
    
    def load_features(self, start_date='20150102', end_date='20150110', window=None, disable_progress=False, how='outer'):
        
        '''
        
        load the features from the raw dataset and return the dict
        
        Parameters
        ----------
        start_date : str
            the starting date of the features, YYYYMMDD format
            
        end_date : str
            the ending date of the features, YYYYMMDD format
            
        window: int
            if you want to load the features in a rolling way,
            enter the rolling window size
        
        disable_progress: bool
            disable the progress bar
            
        how: str in {'inner', 'outer', 'left', 'right'}
            how to merge the raw features. default='outer'
        
        '''
        
        if window is not None:
            # find the first date idx larger than end_date
            end_idx = [idx for idx in range(len(self.dates)) if self.dates[idx] >= end_date][0]
            start_idx = end_idx - window
            if start_idx >= 0:
                start_date = self.dates[start_idx]
        
        
        ret = {}
        
        folders = ['mdv', 'price_volume', 'return', 'risk', 'shout']
        features = [['MDV_63'], 
                    ['CleanMid_open', 'CleanMid_close', 'CumVolume_open', 'CumVolume_close', 'IsOpen_open', 'IsOpen_close'],
                    ['ResidualNoWinsorCumReturn_open', 'ResidualNoWinsorCumReturn_close', 'RawNoWinsorCumReturn_open', 'RawNoWinsorCumReturn_close'],
                    ['estVol'],
                    ['SharesOutstanding']]
        
        for folder_name, feature_list in zip(folders, features):
            for feature_name in feature_list:
                f_name, df = self._load_a_single_feature(start_date, end_date, folder_name, feature_name, how, disable_progress)
                ret[f_name] = df
                
        #Retrieve price from return.
        #Benchmark: the price of all stocks on 20140102 being 100
        price_from_return = (ret['RawNoWinsorCumReturn_close']+1).cumprod()
        price_from_return = 100*price_from_return/price_from_return.iloc[0,:]
        ret['PriceFromReturn_close'] = price_from_return

        #factor 6
        #Intra-day return
        intraday_return = (ret['CleanMid_close']-ret['CleanMid_open'])/ret['CleanMid_open']
        ret['IntradayReturn'] = intraday_return

        #factor 5
        #volatility weighted return
        vol_weighted_return = ret['ResidualNoWinsorCumReturn_close']/ret['estVol']
        vol_weighted_return.dropna(axis=1,how='all').shape
        ret['VolWeightedReturn'] = vol_weighted_return
    
        return ret
    
    
class FeatureLoader:
    
    '''
    load the features from the disk and combine them to be X and y
    
    Parameters
    --------------------
    raw_dir: str
        location of the raw data set
    
    feature_dir: str
        location of the stored feature csv files
        
    target_dir: str
        location of the stored target csv files
        
    pred_dir: str
        location of the prediction files
                
    data: dict of dataframes
        raw data dictionary
    
    '''
    
    def __init__(self, raw_dir=None, feature_dir=None, target_dir=None, pred_dir=None, data=None):
        self.raw_dir = raw_dir.rstrip('/') if raw_dir is not None else None
        self.pred_dir = pred_dir.rstrip('/') if pred_dir is not None else None
        self.feature_dir = feature_dir.rstrip('/') if feature_dir is not None else None
        self.target_dir = target_dir.rstrip('/') if target_dir is not None else None
        self.extractors = []
        self.data = data
        if data is not None:
            self.dates = list(self.data['MDV_63'].index)
        else:
            self.dates = [filename[:-4] for filename in sorted(listdir(self.feature_dir)) if len(filename) == 12]
        
        
    def register(self, extractor_class, name, **kwargs):
        '''
        register a single FeatureExtractor object
        
        Parameters
        ---------------
        extractor_class: callable class name
            the class name of your extractor class
            
        name: str
            the feature name
            
        kwargs: dict
            the optional keyword arguments of your feature extractor
        '''
        extractor = extractor_class(feature_name=name, 
                                    input_dir=self.raw_dir, 
                                    output_dir=self.feature_dir, 
                                    **kwargs).feed(self.data)
        self.extractors.append(extractor)
        
    def extract(self, start_date, end_date, save=True):
        '''
        between start date and end date, 
        extract all the features registered
        
        Parameters
        --------------
        start_date: str 
            date in YYYYMMDD format
        
        end_date: str 
            date in YYYYMMDD format
        '''
        for extractor in self.extractors:
            extractor.extract(start_date=start_date, end_date=end_date, save=save)
        
    
    def combine_a_date(self, date, contain_y=True):
        
        '''
        load a single date's feature and target and combine them.
        Notice that the Ids of feature and target may vary.
        '''
        
        date_idx = self.dates.index(date)
        # read the feature dataframe from disk
        df_X = pd.read_csv(self.feature_dir+'/{:s}.csv'.format(date))
        if contain_y == False:
            return df_X
        # read the next day's closed target dataframe from disk
        df_y = pd.read_csv(self.target_dir+'/{:s}.csv'.format(self.dates[date_idx+1]))
        df_y = df_y.loc[df_y['Time']=='16:00:00.000'][['Id', 'ResidualNoWinsorCumReturn']]
        # inner join the two
        df_tmp = df_X.set_index('Id').join(df_y.set_index('Id'), how='inner')
        df_tmp['Date'] = date

        return df_tmp.reset_index()
    
    def load_features(self, start_date, end_date, disable_progress=False):
        '''
        load the features between start_date and end_date from the disk,
        then concat the dataframes
        '''
        dfs = []
        dates = [date for date in self.dates if (date >= start_date) and (date <= end_date)]
        loop = tqdm(dates, disable=disable_progress, leave=False)
        for date in loop:
            
            df = self.combine_a_date(date)
            dfs.append(df)
            loop.set_description('Loading features')
            
        df_X_y = pd.concat(dfs, axis=0, ignore_index=True)
        df_X_y.sort_values(['Id', 'Date'], ascending=(True, True), inplace=True)
        
        return df_X_y
    
    def load_and_predict(self, model, start_date, end_date, disable_progress=False):
        
        '''
        load the feature csv of each day between start_date and end_date
        and predict the result
        '''
        
        if os.path.isdir(self.pred_dir) == False:
            print('directory {:s} does not exist, creating one'.format(self.pred_dir))
            os.makedirs(self.pred_dir)
            
        dates = [date for date in self.dates if (date >= start_date) and (date <= end_date)]
        loop = tqdm(dates, disable=disable_progress, leave=False)
        for date in loop:
            
            df_X_y = self.combine_a_date(date, contain_y=False)
            df_X = df_X_y.drop(columns=['Id'])
            df_y_pred = pd.DataFrame({'Id': df_X_y['Id'], 
                                      'ResidualNoWinsorCumReturn': model.predict(df_X)})
            df_y_pred.to_csv(self.pred_dir+'/'+date+'.csv', index=False)
            loop.set_description('Generating prediction')
        