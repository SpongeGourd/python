# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:51:37 2016

@author: n.li
"""

import numpy as np
import pandas as pd
from sklearn import metrics

def load():
    df_train = pd.read_csv("D:/LN/Code/Python/majia/tmp_lina_mj_train.csv",sep=',')
    df_test = pd.read_csv("D:/LN/Code/Python/majia/tmp_lina_mj_test.csv",sep=',')
    return df_train,df_test

def preprocess(df_train,df_test):
    #delete non-train columns
    df_train=df_train.drop(['orderid','d','uid'], 1)
    df_test=df_test.drop(['orderid','d','uid'], 1)
    #delete repeat columns
    df_train=df_train.drop(['orderworkdayratio','substitudeorderratio','offlineratio','ppratio'], 1)
    df_test=df_test.drop(['orderworkdayratio','substitudeorderratio','offlineratio','ppratio'], 1)       
    #transform Y to 0or1
    df_train.loc[df_train["orderclass"]=='origin','orderclass']=0
    df_train.loc[df_train["orderclass"]=='mj','orderclass']=1
    df_test.loc[df_test["orderclass"]=='origin','orderclass']=0
    df_test.loc[df_test["orderclass"]=='mj','orderclass']=1
    #transform \N and null to Nan and string to float
    df_train.replace('\N',np.nan,inplace=True)
    df_train.replace('null',np.nan,inplace=True)
    df_train=df_train.astype(float)
    df_test.replace('\N',np.nan,inplace=True)
    df_test.replace('null',np.nan,inplace=True)
    df_test=df_test.astype(float)
    #fill missing data
    df_train.fillna(df_train.mean(),inplace=True)
    df_test.fillna(df_test.mean(),inplace=True)    
    return df_train,df_test    
    
def dfxy(df_train,df_test):
    #define Y and X
    y_train = df_train["orderclass"]
    y_test = df_test["orderclass"]
    x_train = df_train.copy()
    x_test = df_test.copy()
    del x_train["orderclass"]
    del x_test["orderclass"]    
    return x_train, y_train, x_test, y_test


from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
def L1based_feature_selection(x_train,y_train,x_test):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    #x_train_new = model.transform(x_train)
    coef=model.estimator.coef_
    coef_df = pd.DataFrame(data=coef.ravel(),columns=['coef'])
    coef_df["feature"] = x_train.columns.tolist()
    feature_del = coef_df.loc[coef_df["coef"].abs()<0.00001]
    print 'features to del are :'
    print feature_del
    feature_del_namelist = feature_del.loc[:,'feature'].tolist()
    x_train = x_train.drop(feature_del_namelist, 1)
    x_test = x_test.drop(feature_del_namelist, 1)
    return x_train,x_test



#######################################Model###################################
from sklearn.linear_model import LogisticRegression
def LR(x_train,y_train,x_test,y_test):
    #####LR
    clf = LogisticRegression()
    clf.fit(x_train,y_train)
    predict_y = clf.predict(x_test)
    auc_score=metrics.roc_auc_score(y_test,predict_y)
    print 'LR auc_score=',auc_score
    print metrics.precision_recall_fscore_support(y_test,predict_y)
    w=clf.coef_
    #intercept=clf.intercept_
    w_df = pd.DataFrame(data=w.ravel(),columns=['w'])
    w_df["feature"] = x_train.columns.tolist()
    print 'The w of features are:'    
    print w_df    
    return auc_score

from sklearn.naive_bayes import GaussianNB
def NBgauss(x_train,y_train,x_test,y_test):
    ####Naive Bayes (Gaussian likelihood)
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    predict_y = clf.predict(x_test)
    auc_score=metrics.roc_auc_score(y_test,predict_y)
    print 'GaussianNB auc_score=',auc_score
    print metrics.precision_recall_fscore_support(y_test,predict_y)
    return auc_score

from sklearn.ensemble import GradientBoostingClassifier
def GBDT(x_train,y_train,x_test,y_test):
    #####GBDT
    clf = GradientBoostingClassifier()
    clf.fit(x_train,y_train)
    predict_y = clf.predict(x_test)
    auc_score=metrics.roc_auc_score(y_test,predict_y)
    print 'GBDT auc_score=',auc_score
    print metrics.precision_recall_fscore_support(y_test,predict_y)
    return auc_score

from sklearn.ensemble import AdaBoostClassifier 
def AdaBoost(x_train,y_train,x_test,y_test):
    #####AdaBoostClassifier
    clf = AdaBoostClassifier()
    clf.fit(x_train,y_train)
    predict_y = clf.predict(x_test)
    auc_score=metrics.roc_auc_score(y_test,predict_y)
    print 'AdaBoost auc_score=',auc_score
    print metrics.precision_recall_fscore_support(y_test,predict_y)
    return auc_score

from sklearn.ensemble import RandomForestClassifier
def RF(x_train,y_train,x_test,y_test):
    #####RF
    clf = RandomForestClassifier()
    clf.fit(x_train,y_train)
    predict_y = clf.predict(x_test)
    auc_score=metrics.roc_auc_score(y_test,predict_y)
    print 'RF auc_score=',auc_score
    print metrics.precision_recall_fscore_support(y_test,predict_y)
    return auc_score

from sklearn.tree import DecisionTreeClassifier
#from sklearn.externals.six import StringIO
#from sklearn.tree import export_graphviz
#import pydot
def DecisionTree(x_train,y_train,x_test,y_test):
    #####DecisionTree
    clf = DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    predict_y = clf.predict(x_test)
    auc_score=metrics.roc_auc_score(y_test,predict_y)
    print 'DecisionTree auc_score=',auc_score
    print metrics.precision_recall_fscore_support(y_test,predict_y)    
#    dot_data = StringIO()
#    export_graphviz(clf, out_file=dot_data,max_depth=3,feature_names=x_train.columns.tolist())
#    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#    graph.write_pdf("DecisionTreeplot.pdf") 
    return auc_score



df_train,df_test=load()
df_train,df_test=preprocess(df_train,df_test)
x_train,y_train,x_test,y_test=dfxy(df_train,df_test)

x_train,x_test=L1based_feature_selection(x_train,y_train,x_test)
auc1 = LR(x_train,y_train,x_test,y_test)
auc2 = NBgauss(x_train,y_train,x_test,y_test)
auc3 = GBDT(x_train,y_train,x_test,y_test)
auc4 = AdaBoost(x_train,y_train,x_test,y_test)
auc5 = RF(x_train,y_train,x_test,y_test)
auc6 = DecisionTree(x_train,y_train,x_test,y_test)

#Pearson corr
corr_result=df_train.corr()
corr_result.to_csv("corr_result.csv")



import scipy.special as special

def FPvalue( *args):
    """ Return F an p value

    """
    df_btwn, df_within = __degree_of_freedom_( *args)

    mss_btwn = __ss_between_( *args) / float( df_btwn)   
    mss_within = __ss_within_( *args) / float( df_within)

    F = mss_btwn / mss_within    
    P = special.fdtrc( df_btwn, df_within, F)

    return( F, P)

def EffectSize( *args):
    """ Return the eta squared as the effect size for ANOVA

    """    
    return( float( __ss_between_( *args) / __ss_total_( *args)))

def __concentrate_( *args):
    """ Concentrate input list-like arrays

    """
    v = list( map( np.asarray, args))
    vec = np.hstack( np.concatenate( v))
    return( vec)

def __ss_total_( *args):
    """ Return total of sum of square

    """
    vec = __concentrate_( *args)
    ss_total = sum( (vec - np.mean( vec)) **2)
    return( ss_total)

def __ss_between_( *args):
    """ Return between-subject sum of squares

    """    
    # grand mean
    grand_mean = np.mean( __concentrate_( *args))

    ss_btwn = 0
    for a in args:
        ss_btwn += ( len(a) * ( np.mean( a) - grand_mean) **2)

    return( ss_btwn)

def __ss_within_( *args):
    """Return within-subject sum of squares

    """
    return( __ss_total_( *args) - __ss_between_( *args))

def __degree_of_freedom_( *args):
    """Return degree of freedom

       Output-
              Between-subject dof, within-subject dof
    """   
    args = list( map( np.asarray, args))
    # number of groups minus 1
    df_btwn = len( args) - 1

    # total number of samples minus number of groups
    df_within = len( __concentrate_( *args)) - df_btwn - 1

    return( df_btwn, df_within)







