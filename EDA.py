from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
import numpy as np
lowess = sm.nonparametric.lowess

class EDA:
    
    '''
    class to do the exploratory data analysis
    '''
    
    def __init__(self, df):
        self.df = df
        self.size = len(df)
        
    def bin_plot(self, x_label, y_label, n_bins, w_label=None, scale=1, ax=None, **plot_kwargs):
        
        '''
        create a bin plot with your feature (x_label), binned into (num_bin) quantiles 
        (with equal count), on the X-axis, and the corresponding mean of the target
        variable (y_label) along with one standard error around mean on the Y-axis. 
        
        Parameters
        ----------------
        x_label: str
            label of feature x
            
        y_label: str
            label of target
            
        w_label: str
            label of weight if provided
            
        scale: float, default=1
            where to plot the point with the bin = mean(bin) * scale
            
        ax: axes
            ax to plot on, if None, the function will create one
            
        n_bins: int
            number of bins
            
        **plot_kwargs: dict
            keyword and arguments of ax.plot(x, y)
        '''
        
        key = self.df[x_label].values
        value = self.df[y_label].values
        if w_label is not None:
            w = self.df[w_label].values
        
            
        sorted_idx = np.argsort(key)
        chunks = np.array_split(sorted_idx, n_bins)
              
        # x axis on the plot: bin mean
        x = np.ones(n_bins)
        # boundary of the bins
        b = np.ones(n_bins)
        # y axis on the plot: bin mean
        y = np.ones(n_bins)
        # confidence of y: bin std
        c = np.ones(n_bins)
        
        for bin_idx, idx in enumerate(chunks):
            
            x[bin_idx] = np.mean(key[idx]) * scale
            b[bin_idx] = np.max(key[idx])
            if w_label is not None:
                y[bin_idx] = np.average(value[idx], weights=w[idx])
            else:
                y[bin_idx] = np.mean(value[idx])
            c[bin_idx] = np.std(value[idx]) / np.sqrt(len(value[idx]))
        
        if ax is None: 
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ymin = np.min(y-c) - 0.5 * abs(np.min(y-c))
        ymax = np.max(y+c) + 0.5 * abs(np.max(y-c))
            
        ax.plot(x, y, 'o-', **plot_kwargs)
        ax.fill_between(x, y-c, y+c, color='b', alpha=0.1)
        ax.vlines(b[:-1], ymin=ymin, ymax=ymax, color='grey')
        ax.set_xticks(b[:-1])
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlabel('bin boundaries, based on quantiles of {:s}'.format(x_label))
        ax.set_ylabel('within-bin means and 1 std error of {:s}'.format(y_label))
        ax.set_title('{:d} binned plot of {:s} against {:s}'.format(n_bins, y_label, x_label))
        
        
    def scatter(self, x_label, y_label, ax=None, **plot_kwargs):
        
        '''
        plot the ordinary scatter plot
        
        Parameters
        ----------------
        x_label: str
            label of feature x
            
        y_label: str
            label of target
            
        ax: axes
            ax to plot on, if None, the function will create one
            
        **plot_kwargs: dict
            keyword and arguments of ax.scatter(x, y)
        
        '''
        
        x = self.df[x_label]
        y = self.df[y_label]
        
        if ax is None:
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.scatter(x, y, **plot_kwargs)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Scatter plot of {:s} against {:s}'.format(y_label, x_label))
        
    def lowess_plot(self, x_label, y_label, ax=None, sample_size=1.0, frac=1./3, **plot_kwargs):
        
        '''
        
        plot the scatter plot with lowess plot and the zoomed window
        
        A continuous version of binned plot
        
        see https://en.wikipedia.org/wiki/Local_regression for detail of lowess
        
        Parameters
        ----------------
        x_label: str
            label of feature x
            
        y_label: str
            label of target
            
        ax: axes
            ax to plot on, if None, the function will create one
            
        frac: float, default=1./3
            fraction parameter of lowess
            
        **plot_kwargs: dict
            keyword and arguments of ax.scatter(x, y)
        '''
        
        df_xy = self.df[[x_label, y_label]].dropna(inplace=False).sample(frac=sample_size)
        
        x = df_xy[x_label].values
        y = df_xy[y_label].values
        
        model = sm.OLS(y,x)
        fitted = model.fit()
        w = fitted.predict(x)
        
        if ax is None: 
            fig, ax = plt.subplots(figsize=(8, 8))
  
        z = lowess(y, x, return_sorted=True, frac=frac)
        ax.scatter(x,y,color='tab:blue', label='scatter', **plot_kwargs)
        ax.plot(z[:,0],z[:,1],color='tab:red', label='lowess')
        ax.plot(x, w, ':', color='black', label='linear regression')
        ax.set_title('lowess of {:s} against {:s}: R$^2$ = {:.4f}%'.format(y_label, x_label, 100*fitted.rsquared))
        ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        axins = inset_axes(ax, width="30%", height=2., loc=2) 
        ymin = np.min(np.append(z[:,1],w))
        ymax = np.max(np.append(z[:,1],w))
        axins.set_ylim([ymin, ymax])
        axins.scatter(x,y,color='tab:blue', label='scatter', **plot_kwargs)
        axins.plot(z[:,0],z[:,1],color='tab:red', label='lowess')
        axins.plot(x, w, ':',color='black', label='linear regression')
        axins.set_xlabel('zoomed')
        # Turn off tick labels
        axins.set_yticklabels([])
        axins.set_xticklabels([])
        
