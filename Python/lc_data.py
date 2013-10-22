# -*- coding: utf-8 -*-

"""
File: lc_data.py 

This module defines the LC_Data class.
"""

import numpy as np
import pandas as pd
import re
import nltk
import sklearn.feature_extraction as sk_fe

class LC_Data(object):
    """This class represents a Lending Club dataset."""

    def __init__(self):
        self.dataset = pd.DataFrame()
        self._end_file = ''
        self._monthly_file = ''
        self._wd = ''

    def __str__(self):
        """Returns the string representation of the dataset."""
        return "Main data file: " + self._end_file + " " \
        + "Monthly data file: " + self._monthly_file

    def data_import(self, 
                    lc_end_file = "LoanStatsNew.csv", 
                    lc_monthly_file = "Pmthist All Loans Version 20130416.csv",
                    wd = '',
                    index = None):
        
        """Imports Lending Club data and correctly formats and merges it"""

        self._end_file = lc_end_file
        self._monthly_file = lc_monthly_file
        self._wd = wd
        
        if self._wd != '':
            self._wd = re.sub('(.*?)/?$', '\\1/', self._wd)

        """Load data and keep relevant columns"""

        lc_end_rows = pd.read_csv(
            self._wd + self._end_file,
            sep = ',',
            skiprows = 1,
            usecols = ["id"],
            squeeze = True
            ).apply(str)

        lc_end_rows = lc_end_rows.str.contains('\\D|^$', na = True)
        lc_end_rows = lc_end_rows[lc_end_rows == True].index[0]

        lc_end = pd.read_csv(
            self._wd + self._end_file,
            sep = ',',
            na_values = (" ", "N/A", "NULL", "", "?", "n/a", "null"),
            true_values = ("TRUE", "true", "True", "Yes", "yes", "y"),
            false_values = ("FALSE", "false", "False", "No", "no", "n"),
            skiprows = 1,
            nrows = lc_end_rows
            )

        lc_end = lc_end.loc[:,(
            #  "member_id", # unnecessary (non-repeating)
            #  "loan_amnt", # unnecessary
            "funded_amnt", # too late, but needed to calculate ROI
            #  "funded_amnt_inv", #too late
            "term", # filter
            "apr", # keep - interact with int_rate?
            "int_rate", # keep - interact with apr?
            #  "installment", # changes over time
            "grade", # keep
            "sub_grade", # keep
            "emp_name", # keep - nlp
            "emp_length", # keep
            "home_ownership", # keep
            "annual_inc", # keep
            "is_inc_v", # keep
            #  "accept_d", # unnecessary
            #  "exp_d", # unnecessary
            #  "list_d", #unnecessary
            "issue_d", # maybe - worth it to use a date as a predictor?
            "loan_status", # filter, response
            #  "pymnt_plan", # too late
            #  "url", # unnecessary
            "desc", # keep - nlp
            "purpose", # keep - nlp
            "title", # keep - nlp
            #  "addr_city", # maybe - match to demographics? worth it?
            "addr_state", # keep
            "acc_now_delinq", # keep
            #  "acc_open_past_24mths", # keep - after 2011
            #  "bc_open_to_buy", # keep - after 2011
            #  "percent_bc_gt_75", # keep - after 2011
            #  "bc_util", # keep - after 2011
            "dti", # keep
            "delinq_2yrs", # keep 
            "delinq_amnt", # keep
            "earliest_cr_line", # keep - subtract from issue_d?
            "fico_range_low", # keep
            #  "fico_range_high", # unnecessary
            "inq_last_6mths", # keep
            "mths_since_last_delinq", # keep
            "mths_since_last_record", # keep
            #  "mths_since_recent_inq", # keep - after 2011
            #  "mths_since_recent_loan_delinq", # keep - after 2011
            #  "mths_since_recent_revol_delinq", # keep - after 2011
            #  "mths_since_recent_bc", # keep - after 2011
            "mort_acc", # keep
            "open_acc", # keep
            #  "pub_rec_gt_100", # keep - after 2011
            "pub_rec", # keep
            #  "total_bal_ex_mort", # keep - after 2011
            #  "revol_bal", # unnecessary
            #  "revol_util", # too late
            #  "total_bc_limit", # keep - after 2011
            "total_acc", # keep
            "initial_list_status", # keep 
            #  "out_prncp", # too late
            #  "out_prncp_inv", # too late
            #  "total_pymnt", # too late 
            #  "total_pymnt_inv", # too late
            #  "total_rec_prncp", # too late
            #  "total_rec_int", # too late 
            #  "total_rec_late_fee", # too late 
            #  "last_pymnt_d", # too late 
            #  "last_pymnt_amnt", # too late 
            #  "next_pymnt_d", # too late 
            #  "last_credit_pull_d", # too late
            #  "last_fico_range_high", # unnecessary
            #  "last_fico_range_low", # unnecessary
            #  "total_il_high_credit_limit", # too late
            #  "mths_since_oldest_il_open", # keep - after 2011
            #  "num_rev_accts", # keep - after 2011
            #  "mths_since_recent_bc_dlq", # keep - after 2011
            "pub_rec_bankruptcies", # keep
            #  "num_accts_ever_120_pd",  # keep - after 2011
            "chargeoff_within_12_mths", # keep
            "collections_12_mths_ex_med", # keep
            "tax_liens", # keep
            #  "mths_since_last_major_derog", # keep - after 2011
            #  "num_sats", # keep - after 2011
            #  "num_tl_op_past_12m", # keep - after 2011
            #  "mo_sin_rcnt_tl", # keep - after 2011
            #  "tot_hi_cred_lim", # keep - after 2011
            #  "tot_cur_bal", # keep - after 2011
            #  "avg_cur_bal", # keep - after 2011
            #  "num_bc_tl", # keep - after 2011
            #  "num_actv_bc_tl", # keep - after 2011
            #  "num_bc_sats", # keep - after 2011
            #  "pct_tl_nvr_dlq", # keep - after 2011
            #  "num_tl_90g_dpd_24m", # keep - after 2011
            #  "num_tl_30dpd", # keep - after 2011
            #  "num_tl_120dpd_2m", # keep - after 2011
            #  "num_il_tl", # keep - after 2011
            #  "mo_sin_old_il_acct", # keep - after 2011
            #  "num_actv_rev_tl", # keep - after 2011
            #  "mo_sin_old_rev_tl_op", # keep - after 2011
            #  "mo_sin_rcnt_rev_tl_op", # keep - after 2011
            #  "total_rev_hi_lim", # keep - after 2011
            #  "num_rev_tl_bal_gt_0", # keep - after 2011
            #  "num_op_rev_tl", # keep - after 2011
            #  "tot_coll_amt" # keep - after 2011
            "id" # identifier - keep but don't include in analysis
        )]

        lc_mon = pd.read_csv(
            self._wd + self._monthly_file,
            sep = ",",
            na_values = ("NA", "N/A", "NULL", "", "?", "n/a", "null"),
            true_values = ("TRUE", "true", "True", "Yes", "yes", "y"),
            false_values = ("FALSE", "false", "False", "No", "no", "n") 
            )

        lc_mon.columns = map(
            lambda x: re.sub("([a-z])([A-Z])", "\\1_\\2", x),
            lc_mon.columns.values)

        lc_mon.columns = map(lambda x: x.lower(), lc_mon.columns.values)

        """
        -----------------------------------------------------------------------
        Organize and Format self._end_file
        -----------------------------------------------------------------------
        """

        """ Custom format dates, percentages, etc. """
        lc_end.issue_d = pd.to_datetime(lc_end.issue_d)
        lc_end.earliest_cr_line = pd.to_datetime(lc_end.earliest_cr_line)
        lc_end.int_rate = lc_end.int_rate.apply(
            lambda x: float(re.sub('[^0-9.]', '', x)) / 100)
        lc_end.apr = lc_end.apr.apply(
            lambda x: float(re.sub('[^0-9.]', '', x)) / 100)
        lc_end.term = lc_end.term.apply(lambda x: int(re.sub('\\D', '', x)))
        lc_end['years_since_earliest_cr_line'] = \
            (lc_end.issue_d - lc_end.earliest_cr_line)
        lc_end.years_since_earliest_cr_line = map(
            lambda x: float(x / (8.64e13)) / 365, # 8.64e13 ns in 1 minute
            lc_end.years_since_earliest_cr_line)

        """ Create data frame of missingness indicators """
        lc_end_missings = lc_end.count(axis = 0)
        lc_end_missings = lc_end_missings[
            lc_end_missings < lc_end.shape[0]].index.values
        lc_end_missings = lc_end.loc[:, lc_end_missings].applymap(pd.isnull)
        lc_end_missings.columns = lc_end_missings.columns.values + "_missing"

        """ Flag columns of different data types """
        lc_end_floatcols = lc_end.dtypes.isin([np.dtype('float64')])
        lc_end_intcols = lc_end.dtypes.isin([np.dtype('int64')])
        lc_end_strcols = lc_end.dtypes.isin([np.dtype('object')])
        lc_end_boolcols = lc_end.dtypes.isin([np.dtype('bool')])
        
        lc_end_floatcols = lc_end_floatcols[lc_end_floatcols].index.values
        lc_end_intcols = lc_end_intcols[lc_end_intcols].index.values
        lc_end_strcols = lc_end_strcols[lc_end_strcols].index.values
        lc_end_boolcols = lc_end_boolcols[lc_end_boolcols].index.values

        """ Specify what value should replace missing values for each data type """
        lc_end_dicts = dict(zip(
            np.concatenate([
                lc_end_floatcols, lc_end_intcols, lc_end_strcols, 
                lc_end_boolcols]),
            ([0.0] * len(lc_end_floatcols)) + ([0] * len(lc_end_intcols)) + 
            (["unk"] * len(lc_end_strcols)) + ([False] * len(lc_end_boolcols))
            ))

        """ Fill missing values and append missingness indicator columns """
        lc_end = pd.concat([lc_end.fillna(lc_end_dicts), lc_end_missings], 
                        axis = 1)

        """ Monthly Data """
        lc_mon['total_paid'] = lc_mon.prncp_paid + lc_mon.fee_paid + lc_mon.int_paid
        lc_mon_payments = lc_mon.loc[:, ('loan_id', 'total_paid')].groupby(
            ['loan_id'], as_index=False).agg(sum)
        lc_mon_payments = lc_mon_payments.sort('loan_id')
        lc_mon_payments['n_months'] = map(float, 
            lc_mon.loan_id.value_counts().sort_index())

        lc_all = pd.merge(
            lc_end, lc_mon_payments, 
            how = 'inner', 
            left_on = 'id', right_on = 'loan_id')
        
        lc_all['return_amount'] = lc_all.total_paid - lc_all.funded_amnt
        lc_all['return_percent'] = lc_all.return_amount / lc_all.funded_amnt

        """ Keep only loans for which we have complete historical records """
        keep_term = lc_all['term'] == 36
        keep_finished = lc_all.loan_status.isin(
            ['Fully Paid', 'Charged Off', 'Default'])
        keep_expired = (lc_all['n_months'] - lc_all['term']) > 0
        remove_educational = lc_all['purpose'] != 'educational'

        lc_all = lc_all[
            keep_term & (keep_finished | keep_expired) & remove_educational]
    
        lc_all = lc_all.drop(["loan_status", "earliest_cr_line", "n_months", 
                          "total_paid", "id"], 1)
    
        lc_all.index = range(lc_all.shape[0])
        
        if index != None:
            self.dataset_index = lc_all.loc[:,index]
        else:
            self.dataset_index = pd.Series(range(lc_all.shape[0]))

        self.dataset = lc_all


    """
    ---------------------------------------------------------------------------
    Function to create term-document matrices for all supplied columns.
    ---------------------------------------------------------------------------
    """
    
    def make_tdms(self, text_fields, count_words = True, 
                  count_percents = True, count_currency = True, set_min_df = 1, 
                  set_ngram_range = (1,1), merge_all = True):
    
        tdms = {}
        
        for texts in text_fields:
            
            if count_words:
                n_words = self.dataset.loc[:,texts].str.count(r'\b\w+\b')
                n_words.index = range(self.dataset.shape[0])
            else:
                n_words = 0
                
            if count_percents:        
                n_percents = self.dataset.loc[:,texts].str.count(
                    r'\d+([.,]\d+)?%')
                n_percents.index = range(self.dataset.shape[0])
            else:
                n_percents = 0
                
            if count_currency:
                n_currency = self.dataset.loc[:,texts].str.count(
                    r'\$\d+|\d+\s+dollar', re.IGNORECASE)
                n_currency.index = range(self.dataset.shape[0])
            else:
                n_currency = 0
            
            if texts == 'desc':
                use_regex = [
                r'(\d+|Borrower)\s+added\s+on\s+(\d{2}/\d{2}/\d{2})\s+>',
                r'(\d+|Borrower)\s+added\s+on\s+']    
            else:
                use_regex = []
            
            cleaned_text = clean_text(self.dataset.loc[:,texts], 
                regex = use_regex)
        
            vectorizer = sk_fe.text.CountVectorizer(
                ngram_range = set_ngram_range, stop_words = 'english', 
                min_df = int(set_min_df))
            tdm = vectorizer.fit_transform(cleaned_text.tolist()).toarray()
            tdm_colnames = np.array(vectorizer.get_feature_names())
            
            transformer = sk_fe.text.TfidfTransformer()
            tdm = transformer.fit_transform(tdm).toarray()
            
            tdm = pd.DataFrame(tdm, columns = tdm_colnames)
            tdm.columns = texts + '__' + tdm.columns.values
            
            if count_words:        
                tdm[texts + '__n_words'] = n_words
        
            if count_percents & sum(n_percents) > 0:
                tdm[texts + '__n_percents'] = n_percents
                
            if count_currency & sum(n_currency) > 0:
                tdm[texts + '__n_currency'] = n_currency
        
            tdms[texts + '_tdm'] = tdm
    
        if(merge_all):
            
            for i in range(len(tdms)):
                self.dataset = pd.concat(
                    [self.dataset, tdms[tdms.keys()[i]]], axis = 1)
            
        else:
            
            self.tdms = tdms

    """
    ---------------------------------------------------------------------------
    Method to convert string variables to dummies
    ---------------------------------------------------------------------------
    """
    def change_to_dummies(self, drop_variables = None):
        if(drop_variables != None):       
            self.dataset_dropped = self.dataset.loc[:, drop_variables]
            self.dataset = self.dataset.drop(drop_variables, axis = 1)
            strcols = self.dataset.dtypes.isin([np.dtype('object')])
            strcols = strcols[strcols].index.values
    
        for col in strcols:
            self.dataset = pd.concat([
                self.dataset, 
                pd.get_dummies(
                    self.dataset.loc[:,col].squeeze(), 
                    prefix = col, prefix_sep = '__')],
                axis = 1)
    
        self.dataset_dropped = pd.concat(
            [self.dataset_dropped, self.dataset.loc[:,strcols]])
        self.dataset = self.dataset.drop(strcols, axis = 1).applymap(float)    


"""
-------------------------------------------------------------------------------
Function to clean a text column for text mining.
-------------------------------------------------------------------------------
"""
    
def clean_text(text, regex = [], remove_punct = True, 
               remove_percents = True, remove_currency = True, 
               remove_numbers = True, to_lower = True, stem = True):

    text_original = text

    if len(regex) == 2:
        
        text = text.str.findall(regex[0])
        text = text[map(lambda x: len(x) > 0, text)]
        text = text.apply(
            lambda x: np.unique(map(lambda y: pd.to_datetime(y[1]), x)))
        text = text.apply(lambda x: np.delete(x, x.argmin()))
        text = text[map(lambda x: len(x) > 0, text)]
        text = text.apply(lambda x: x[0].strftime('%m/%d/%y'))
        
        for i in range(len(text)):
            text_original[text.index[i]] = re.sub(
                regex[1] + text.iloc[i] + '.*', '', 
                text_original[text.index[i]])
        
        text_original = text_original.apply(
            lambda x: re.sub(regex[0], '', x))
        text_original = text_original.apply(
            lambda x: re.sub(r'<.*?>', ' ', x))

    if remove_percents:    
        text_original = text_original.apply(
            lambda x: re.sub(r'\d+([.,]\d+)?%', ' ', x))

    if remove_currency:    
        text_original = text_original.apply(
            lambda x: re.sub(r'\$\d+([.,]\d+)?\b(?!\s+dollar)', ' ', x))
        text_original = text_original.apply(
            lambda x: re.sub(r'\d+([.,]\d+)?\s+dollar', ' ', x))

    if remove_punct:
        text_original = text_original.apply(
            lambda x: re.sub(r'[^\w\s]', ' ', x))

    if remove_numbers:
        text_original = text_original.apply(
            lambda x: re.sub(r'\d', ' ', x))
        
    if to_lower:
        text_original = text_original.str.lower()
    
    if stem:
        text_original = text_original.apply(
            lambda x: ' '.join(map(
                nltk.stem.PorterStemmer().stem, x.lower().split())))
    
    return(text_original)

if __name__ == '__main__':
    d = LC_Data()
    d.data_import()
    d.make_tdms(
        text_fields = ['emp_name', 'desc', 'purpose', 'title'],
        set_min_df = round(d.dataset.shape[0] * 0.005),
        set_ngram_range = (1,4))













