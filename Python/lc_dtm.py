# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Setup
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Load modules and custom functions
#-----------------------------------------------------------------------------#
import os
import pandas as pd
import numpy as np
import re
import nltk
import sklearn.feature_extraction as sk_fe

def clean_text(text, regex = [], remove_punct = True, remove_percents = True,
               remove_currency = True, remove_numbers = True, to_lower = True, 
               stem = True):

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
        
        text_original = text_original.apply(lambda x: re.sub(regex[0], '', x))
        text_original = text_original.apply(lambda x: re.sub(r'<.*?>', ' ', x))

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

# Set working directory (comment out whichever you're not currently using)
#os.chdir('c:/Users/SchaunW/Dropbox/Code/GitHub/LC_Project/Python/') # Windows
os.chdir('/Users/schaunwheeler/Dropbox/Code/GitHub/LC_Project/Python/') # OS X


# Scripts
import lc_data

#-----------------------------------------------------------------------------#
# Workflow
#-----------------------------------------------------------------------------#

# Import baseline data
lc = lc_data.lc_data_import(
    lc_end_file = 'LoanStatsNew.csv',
    lc_monthly_file = 'Pmthist All Loans Version 20130416.csv',
#    wd = 'c:/Users/SchaunW/Dropbox/code/lending_club')
    wd = '/Users/schaunwheeler/Dropbox/code/lending_club')
    
lc_texts = lc.loc[:,('emp_name', 'desc', 'purpose', 'title')]

lc_tdms = {}

for j in range(lc_texts.shape[1]):

    n_words = lc_texts.iloc[:,j].str.count(r'\b\w+\b')
    
    n_percents = map(lambda x: len(re.findall("\\d+([.,]\\d+)?%", x)),
        lc_texts.iloc[:,j])
    n_currency = map(
        lambda x: len(re.findall(
            r'\$\d+([.,]\d+)?\b(?!\s+dollar)', x.lower())) + 
        len(re.findall(r'\d+([.,]\d+)?\s+dollar', x.lower())),
        lc_texts.iloc[:,j])
    
    cleaned_text = clean_text(lc_texts.iloc[:,j], 
        regex = [r'(\d+|Borrower)\s+added\s+on\s+(\d{2}/\d{2}/\d{2})\s+>',
        r'(\d+|Borrower)\s+added\s+on\s+'])

    vectorizer = sk_fe.text.CountVectorizer(ngram_range = (1,1))
    tdm = vectorizer.fit_transform(cleaned_text.tolist()).toarray()
    tdm_keep = (tdm > 0).sum(axis = 0) >= int(round(lc_texts.shape[0] * 0.005))
    tdm_colnames = np.array(vectorizer.get_feature_names())[tdm_keep]
    tdm = tdm[:,tdm_keep]
    
    transformer = sk_fe.text.TfidfTransformer()
    tdm = transformer.fit_transform(tdm).toarray()
    
    tdm = pd.DataFrame(tdm, columns = tdm_colnames)
    tdm.columns = lc_texts.columns.values[j] + '__' + tdm.columns.values
    tdm[lc_texts.columns.values[j] + '__n_words'] = n_words
    tdm[lc_texts.columns.values[j] + '__n_percents'] = n_percents
    tdm[lc_texts.columns.values[j] + '__n_currency'] = n_currency

    lc_tdms[lc_texts.columns.values[j] + '_tdm'] = tdm

























    
    
if __name__ == '__main__':
    lc_data_import()