import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def get_IC(df_X_y, feature_name, target_name='ResidualNoWinsorCumReturn', NA_thresh=100, binplot=True):
    
    '''
        daily IC = daily cross-sectional corr(feature, target)
        IC = mean(daily IC)
        IR = mean(daily IC) / std(daily IC)
        
        from df_X_y:
            1. compute IC & IR
            2. bin-plot of IC distribution, by year
        
        Input:
            df_X_y: dataframe
                dataframe of features & targets, extracted from 'features' folder
            feature_name: str
                feature name in df_X_y columns 
            target_name: str
                target name in df_X_y columns, default 'ResidualNoWinsorCumReturn'
            NA_thresh: int / None
                drop a stock symbol, if ≥ NA_thresh nan values in feature, default 100
                if None, no dropna()
            binplot: bool
                if True, plot bin-plot of IC distribution by year
                
        Output:
            IC mean & std, by year
    '''
    
    # feature
    if feature_name not in df_X_y.columns:
        print('Feature does not exist!')
        return
    feature = df_X_y[feature_name]
    feature.index = pd.MultiIndex.from_frame(df_X_y[['Date','Id']])
    feature = feature.unstack()
    feature.index = pd.to_datetime(feature.index)
    if NA_thresh is not None:
        feature.dropna(axis='columns', thresh=len(feature)-100, inplace=True)    # drop a stock symbol, if ≥ NA_thresh nan values in feature

    # target
    target = df_X_y[target_name]
    target.index = pd.MultiIndex.from_frame(df_X_y[['Date','Id']])
    target = target.unstack()
    target.index = pd.to_datetime(target.index)
    if NA_thresh is not None:
        target = target[feature.columns]                                         # drop a stock symbol, if ≥ NA_thresh nan values in feature

    # IC
    IC = (feature * target).mean(axis='columns') - feature.mean(axis='columns') * target.mean(axis='columns')
    IC /= feature.std(axis='columns')
    IC /= target.std(axis='columns')
    IC = IC.to_frame()
    IC['year'] = IC.index.year
    IC.columns = ['IC', 'year']

    # IC mean & standard error, by year
    IC_summary = pd.concat([IC['IC'].groupby(IC.index.year).mean(),
                           IC['IC'].groupby(IC.index.year).std()], axis='columns')
    IC_summary.columns = ['mean', 'std']
    print(feature_name, ' IC: ', IC_summary['mean'].mean())
    print(feature_name, ' IR: ', IC_summary['mean'].mean() / IC_summary['std'].mean())
    
    # bin-plot of IC distribution, by year
    if binplot == True:
        IC_year = IC_summary.index
        IC_mean = IC_summary['mean']
        IC_std = IC_summary['std']
        plt.plot(IC_year, IC_mean, color='b')
        plt.fill_between(IC_year, IC_mean - IC_std, IC_mean + IC_std, color='b', alpha=0.1)
        plt.errorbar(IC_year, IC_mean, yerr=IC_std, fmt='o', capsize=5, color='b')
        plt.grid()
        plt.show()

    return IC_summary




def get_rank_IC(df_X_y, feature_name, target_name='ResidualNoWinsorCumReturn', NA_thresh=100, binplot=True):
    
    '''
        daily rank_IC = daily cross-sectional corr(rank(feature), rank(target))
        rank_IC = mean(daily rank_IC)
        rank_IR = mean(daily rank_IC) / std(daily rank_IC)
        
        from df_X_y:
            1. compute rank_IC & rank_IR
            2. bin-plot of IC distribution, by year
        
        Input:
            df_X_y: dataframe
                dataframe of features & targets, extracted from 'features' folder
            feature_name: str
                feature name in df_X_y columns 
            target_name: str
                target name in df_X_y columns, default 'ResidualNoWinsorCumReturn'
            NA_thresh: int / None
                drop a stock symbol, if ≥ NA_thresh nan values in feature, default 100
                if None, no dropna()
            binplot: bool
                if True, plot bin-plot of IC distribution by year
                
        Output:
            rank_IC mean & std, by year
    '''
    
    # feature
    if feature_name not in df_X_y.columns:
        print('Feature does not exist!')
        return
    feature = df_X_y[feature_name]
    feature.index = pd.MultiIndex.from_frame(df_X_y[['Date','Id']])
    feature = feature.unstack()
    feature.index = pd.to_datetime(feature.index)
    if NA_thresh is not None:
        feature.dropna(axis='columns', thresh=len(feature)-100, inplace=True)    # drop a stock symbol, if ≥ NA_thresh nan values in feature

    # feature rank
    feature_rank = feature.rank(axis='columns')
    
    # target
    target = df_X_y[target_name]
    target.index = pd.MultiIndex.from_frame(df_X_y[['Date','Id']])
    target = target.unstack()
    target.index = pd.to_datetime(target.index)
    if NA_thresh is not None:
        target = target[feature.columns]                                         # drop a stock symbol, if ≥ NA_thresh nan values in feature
    
    # target rank
    target_rank = target.rank(axis='columns')

    # IC
    IC = (feature_rank * target_rank).mean(axis='columns') - feature_rank.mean(axis='columns') * target_rank.mean(axis='columns')
    IC /= feature_rank.std(axis='columns')
    IC /= target_rank.std(axis='columns')
    IC = IC.to_frame()
    IC['year'] = IC.index.year
    IC.columns = ['IC', 'year']

    # IC mean & standard error, by year
    IC_summary = pd.concat([IC['IC'].groupby(IC.index.year).mean(),
                           IC['IC'].groupby(IC.index.year).std()], axis='columns')
    IC_summary.columns = ['mean', 'std']
    print(feature_name, ' IC: ', IC_summary['mean'].mean())
    print(feature_name, ' IR: ', IC_summary['mean'].mean() / IC_summary['std'].mean())
    
    # bin-plot of IC distribution, by year
    if binplot == True:
        IC_year = IC_summary.index
        IC_mean = IC_summary['mean']
        IC_std = IC_summary['std']
        plt.plot(IC_year, IC_mean, color='b')
        plt.fill_between(IC_year, IC_mean - IC_std, IC_mean + IC_std, color='b', alpha=0.1)
        plt.errorbar(IC_year, IC_mean, yerr=IC_std, fmt='o', capsize=5, color='b')
        plt.grid()
        plt.show()

    return IC_summary