# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:59:47 2016

@author: yihongchen
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as ss


def _check_discrete(X):
    """
    try to check the valus is discrete or not 
    """
    unique_x = np.unique(X)
    percent = unique_x.shape[0]/X.shape[0]
    if(percent>=0.1):
        return False
    else:
        return True
        
        
def _check_dataFrame(X):
    import warnings
    if(isinstance(X,pd.DataFrame)!=True):
        warnings.warn("Data is not a DataFrame")
        X = pd.DataFrame(data=X)
        return X
    return X        
    
def missingFeature(df):
    """
    plot the missing percent for every dataframe
    """
    from preprocess import checkNaPos
    columns = df.columns
    df_missing = pd.DataFrame(data  = np.zeros((columns.ravel().shape[0],2)),columns=['columns','value'])
    df_missing['columns'] = columns
    for i_column in xrange(columns.shape[0]):
        na_percent = checkNaPos(df[columns[i_column]])
        df_missing.iloc[i_column,1] = na_percent
    df_missing.sort(columns='value',inplace=True)
    return df_missing
 

   
def explore_na(df,all_columns=True):
    """
    explore the na percent and the feature
    """
    df_missing_percent = missingFeature(df)
    if(all_columns==False):
        df_missing_percent = df_missing_percent[df_missing_percent['value']>=0.000001]
    f,ax = plt.subplots()
    ind = np.arange(df_missing_percent['value'].shape[0])
    width  = 0.3
    rects1 = ax.bar(ind, (df_missing_percent['value']*100).astype(int),width) 
    
    ax.set_xticks(ind+width)
    ax.set_xticklabels(df_missing_percent['columns'].values, rotation='vertical')
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    return [f],df_missing_percent   


class Explore_Distribution():
    def __init__(self):
        self.fig_list =[]
    
    def plot(self,X,kind='hist',subPlots=True):
        X =  _check_dataFrame(X)
        self.fig_list = []
        if(subPlots!=True):
            fig = X.plot(kind=kind)
            self.fig_list.append(fig)
        else:
            #plot in different figure
            fig = plt.figure()
            for i in xrange(X.shape[1]):
                df = X[[X.columns[i]]]
                df.plot(kind=kind,subplots=True)
                self.fig_list.append(fig)
        
        return self.fig_list
    
    def summary(self,X):
        """
        
        """
        pass




def Similary_Matrix(X,y,strategy='pearson'):
    """
    calculate the similary matrix for different type  
    """
    import warnings
    
    if(strategy=='pearson'):
        if(_check_discrete(X)|_check_discrete(y)):
            warnings.warn('pearson used in Discrete data type')
            cal_strategy = ss.pearsonr(X,y)
    elif(strategy==''):
        pass
        
 
class Explore_Connection():
    def __init__(self):
        self.fig_list =[]
        
    def plot(self,X,y,kind='scatter',subPlots=True):
        X =  _check_dataFrame(X)
        if(subPlots!=True):
            fig = plt.plt(X,y)
            self.fig_list.append(fig)
        else:
            #plot in different figure
            fig = plt.figure()
            for i in xrange(X.shape[1]):
                df = X[[X.columns[i]]]
                plt.plot(df,y,"*")
                self.fig_list.append(fig)
        
        
    
    def Similary_Matrix(self,X,y=None):
        """
        calculate the 
        """
        pass
        
                
        
    def histgram_compare(self,X,y,label=1,nbins = 20):
        """
        plot the data in histgram which in the 
        """
        featureNames = X.columns
        self.fig_list = [] 
        for c in  featureNames:
            try:
                if X[c].dtype != 'object':
                    fig=plt.figure(figsize=(14,4))
                    ax1 = fig.add_subplot(1,2,1) 
                    
                    dataset1 = X[c][~np.isnan(X[c])& (y!=label)]
                    dataset2 = X[c][~np.isnan(X[c]) & (y==label)]
                    # left plot
                    hd = ax1.hist((dataset1, dataset2), bins=nbins, histtype='bar',normed=True,
                                    color=["blue", "red"],label=['all','target=1'])
                    ax1.set_xlabel('Feature: %s'%c)
                    ax1.set_xlim((-1,max(X[c])))
                    
                    binwidth = hd[1][1]-hd[1][0]
                    midpts = (hd[1][:-1]+hd[1][1:])/2
                    cdf_all= np.cumsum(hd[0][0])*binwidth
                    cdf_ones = np.cumsum(hd[0][1])*binwidth
            
                    # right plot
                    ax2 = fig.add_subplot(1,2,2) 
                    ax2.set_ylim((0,1))
                    ax2.set_xlim((0,nbins))
                    ax2.plot(midpts,cdf_all,color='b')
                    ax2.plot(midpts,cdf_ones,color='r')
                    ax2.plot(midpts,0.5+10*(cdf_all-cdf_ones),color='k')
                    ax2.grid()
                    ax2.set_xlim((-1,max(X[c])))
                    ax2.set_xlabel('cdfs plus cdf_diff*10+0.5')
                    ax2.axhline(0.5,color='gray',linestyle='--')
                    
                    self.fig_list.append(fig)
            except:
                print 'error%s'%c
        plt.show()


        
        
        
    
        
if __name__=='__main__':
    from sklearn.datasets import make_classification,make_regression
    df_x,df_y = make_classification(n_samples =10000,n_features=20,n_classes = 10,n_informative=10)
    dr_x,dr_y = make_regression(n_samples =1000,n_features=3)
    df_x=_check_dataFrame(df_x)
    dr_x=_check_dataFrame(dr_x)
    
    ed = Explore_Distribution()
    
    ed.plot(dr_x,kind='box',subPlots=True)    
    
    plot_class = Explore_Connection()
    
    plot_class.plot(dr_x,dr_y)
    
    from ModelChoosen import ClassifierModel
    
    clf = ClassifierModel()
    clf.fit(df_x,df_y)
