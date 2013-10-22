# -*- coding: utf-8 -*-

import sklearn.externals as sk_ex
import sklearn.ensemble as sk_en
import sklearn.cross_validation as sk_cv
import warnings
import numpy as np
import scipy as sp
import pandas as pd

rfr = sk_en.RandomForestRegressor(
    n_estimators = 300, 
    oob_score = True,
    compute_importances = True,
    min_samples_split = 10,
    verbose = 2,
    n_jobs = 4,
    random_state = 42).fit(
        X = lc_x.drop(['return_percent'], axis = 1),
        y = lc_x.return_percent)
    
lc_rf_imp = pd.Series(
    rfr.feature_importances_, 
    index =lc_x.drop(['return_percent'], axis = 1).columns.values)
lc_rf_imp = lc_rf_imp[lc_rf_imp != 0].order(ascending = False)

rfrs = [rfr]

rf_ind = np.array(map(
    lambda x: int(round(np.log(x))), 
    range(1, len(lc_rf_imp) + 1)))

rf_ind_counter = np.unique(rf_ind)[::-1]

for x in rf_ind_counter[1:]:
    new_cols = lc_rf_imp.index.values[rf_ind <= x]
    
    rfr_new = sk_en.RandomForestRegressor(
        n_estimators = 300, 
        oob_score = True,
        compute_importances = True,
        min_samples_split = 10,
        verbose = 2,
        n_jobs = 4,
        random_state = 42).fit(
            X = lc_x.loc[:, new_cols],
            y = lc_x.return_percent)
        
    rfr_new.feature_importances_ = \
        pd.Series(rfr_new.feature_importances_, index = new_cols)
    
    rfrs.append(rfr_new)

# would be useful to do a more in-depth search for optimum n features
    
rfr_keep = rfrs[np.array(map(lambda x: x.oob_score_, rfrs)).argmax()]

_ = sk_ex.joblib.dump(rfr_keep, 'Model Outputs/rfr_keep.pkl') 

# determined by previous analysis
rfr_keep = sk_ex.joblib.load('Model Outputs/rfr_keep.pkl')

lc['return_percent_prediction'] = rfr_keep.oob_prediction_
    
lc_target = lc[lc.return_percent_prediction > 0.0]
#lc_target['rpp_round'] = lc_target.return_percent_prediction.round(2)

lc_cuts = np.array([0.0, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.125, 0.135, 
                    0.145, 0.155, 1.0])

lc_target['return_percent_cut1'] = pd.cut(lc_target.return_percent_prediction, 
    bins = lc_cuts, 
    labels = np.arange(0, 251, 25))
lc_target.return_percent_cut1.value_counts()[lc_target.return_percent_cut1.value_counts().index.order()]

lc_target['return_percent_cut2'] = pd.cut(lc_target.return_percent_prediction, 
    bins = 11, labels = np.arange(0, 251, 25))
lc_target.return_percent_cut2.value_counts()[lc_target.return_percent_cut2.value_counts().index.order()]

def bootstrap_roi(
    df, bin_variable, roi_variable, ndollars = 1000, nboot = 1000):

    output = []
    
    for i in range(nboot):    
        boot_ind = np.random.choice(df.index.values, np.ceil(ndollars/25))
        boot_ind_subset = (df.loc[boot_ind][bin_variable]).cumsum() 

        if list(boot_ind_subset)[-1] < ndollars:
            warnings.warn('Not enough opportunites to spend all dollars.')
            
        boot_ind = boot_ind[boot_ind_subset < ndollars]

        profit = sum(
            df[roi_variable][boot_ind] * df[bin_variable][boot_ind])
            
        output.append(profit / ndollars)
        
    return(pd.Series(output))

bootstrap_roi(
    lc_target, 
    bin_variable = 'return_percent_cut1', roi_variable = 'return_percent', 
    ndollars = 10000, nboot = 1000).describe(95)


# find optimal cut points for predictions
# run separate random forests and average results
# gradient boosting
# naive bayes
# svm
# knn



sk_cv.train_test_split
sk_cv.KFold

