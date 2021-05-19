# standard packages
from os import listdir
from tqdm import tqdm
import os
import dill as pickle
import argparse
# data science stack packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
# our implementation
from dataloader import DataLoader, FeatureLoader
from features import *
from selection import GridSearchWeightedCV
from selection import SelectKBestIC, SelectKBestWeightedR2
from pipeline import Winsorizer
from EDA import EDA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('input arguments taking the following command line options')
parser.add_argument('-i', help='input directory of your raw data (mode 1) or feature (mode 2)', type=str)
parser.add_argument('-o', help='output directory of your feature (mode 1) or predictedion (mode 2)', type=str)
parser.add_argument('-p', help='directory containing the learned model/s you provided (mode 2)', type=str)
parser.add_argument('-s', help='start date in YYYYMMDD format', type=str)
parser.add_argument('-e', help='end date in YYYYMMDD format', type=str)
parser.add_argument('-m', help='mode you will run in. support two modes', type=int)

def mode_1(args):
    
    '''
    mode 1, extract features from raw data
    '''
    
    max_lag = 21
    dataloader = DataLoader(filepath=args.i)
    # load the raw data
    start_date_idx = dataloader.dates.index(args.s)
    if start_date_idx - max_lag >= 0:
        start_date = dataloader.dates[start_date_idx - max_lag]
    else:
        start_date = '20140102'
    print('loading {:s} from {:s} to {:s}'.format(args.i, start_date, args.e))
    data = dataloader.load_features(
                            start_date=start_date, 
                            end_date=args.e, 
                            how='outer')
    
    # load the feature from the saved csv
    if os.path.isdir(args.o) == False:
        print('directory {:s} does not exist, creating one'.format(args.o))
        os.makedirs(args.o)
    feature_loader = FeatureLoader(raw_dir=args.i, # the raw dataset 
                                   feature_dir=args.o, # where to save the features
                                   target_dir=args.i+'/return', # your target directory
                                   data=data)
    
    # first you register your feature extractor into the featureloader
    # remember to pass the keyword arguments of your extractor

    # Jordan's features
    feature_loader.register(X_weighted_Y, 
                            name='ts_sum(res_ret*vol,20)/ts_sum(vol,20)',
                            X_name='CumVolume_close', 
                            Y_name='ResidualNoWinsorCumReturn_close', N=20)
    feature_loader.register(corr_X_Y, 
                            name='corr(vol,raw_ret,20)',
                            X_name='CumVolume_close', 
                            Y_name='RawNoWinsorCumReturn_close', N=20)
    feature_loader.register(ts_kurt_X, 
                            name='ts_kurt(raw_ret,20)', 
                            X_name='RawNoWinsorCumReturn_close', N=20)
    feature_loader.register(ts_sum_X, 
                            name='ts_sum(res_ret,5)', 
                            X_name='ResidualNoWinsorCumReturn_close', N=20)
    feature_loader.register(MQ, 
                            name='MQ(res_ret,10,0.75,0.25)', 
                            X_name='ResidualNoWinsorCumReturn_close', 
                            N=10, X_rank=False, first_quantile=0.75, second_quantile=0.25)
    feature_loader.register(MQ, 
                            name='MQ(res_ret,20,0.75,0.25)', 
                            X_name='ResidualNoWinsorCumReturn_close', 
                            N=20, X_rank=False, first_quantile=0.75, second_quantile=0.25)
    feature_loader.register(ts_sum_X_Y, 
                            name='ts_sum(res_ret_day,20)', 
                            X_name='ResidualNoWinsorCumReturn_close',
                            Y_name='ResidualNoWinsorCumReturn_open', 
                            N=20, __and__=np.subtract)
    feature_loader.register(ts_sum_X, 
                            name='ts_sum(raw_ret,5)', 
                            X_name='RawNoWinsorCumReturn_close', N=5)
    feature_loader.register(MQ, 
                            name='MQ(raw_ret,20,0.75,0.25)',
                            X_name='RawNoWinsorCumReturn_close', 
                            N=20, X_rank=False, first_quantile=0.75, second_quantile=0.25)
    feature_loader.register(ts_kurt_X, 
                            name='ts_kurt(rank(estVol),20)', 
                            X_name='estVol', N=20, X_rank=True)
    feature_loader.register(ts_kurt_X, 
                            name='rank(ts_kurt(vol,20))',
                            X_name='CumVolume_close', N=20, feature_rank=True)
    
    # Kaylie's features
    feature_loader.register(Squared, 
                            name='RawReturn_LogLogSquared', 
                            input_variable='RawNoWinsorCumReturn_close')
    feature_loader.register(Reversal, 
                            name='Reversal1D', window=0)
    feature_loader.register(Reversal, 
                            name='Reversal5D', window=4)
    feature_loader.register(Liquidity, 
                            name='Liquidity_volume_close', 
                            input_variable_name='CumVolume_close', window=5)
    feature_loader.register(DataNdaysBefore, 
                            name='IntradayReturn', 
                            input_variable_name='IntradayReturn', window=0)
    feature_loader.register(FD, 
                            name='FD1008', 
                            input_variable='PriceFromReturn_close', window=10, d=0.8)
    feature_loader.register(BB, 
                            name='BB', input_variable_name='PriceFromReturn_close', 
                            window=20,k=2)
    feature_loader.register(DataNdaysBefore, 
                            name='RawNoWinsorCumReturn_close', 
                            input_variable_name='RawNoWinsorCumReturn_close', window=0)
    
    print('extracting features {:s}-{:s} to {:s}'.format(args.s, args.e, args.o))
    feature_loader.extract(start_date=args.s, end_date=args.e, save=True);
    print('mode 1 completed')
    
    
def mode_2(args):
    
    '''
    mode 2, make predictions
    '''
    
    print('loading the fitted model pickle file')
    model = pickle.load(open(args.p, 'rb'))
    feature_loader = FeatureLoader(feature_dir=args.i, pred_dir=args.o)
    print('predicting')
    feature_loader.load_and_predict(model, start_date=args.s, end_date=args.e)
    print('mode 2 completed')

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    if args.m == 1:
        mode_1(args)
    elif args.m == 2:
        mode_2(args)
    else:
        print('mode has to be either 1 or 2')