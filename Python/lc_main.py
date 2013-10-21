# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Setup
#-----------------------------------------------------------------------------#

# All dependencies for alll scripts (for reference - we'll delete this later)
import os
import pandas as pd
#import numpy as np
#import re
#import nltk
#import sklearn.feature_extraction as sk_fe

# Set working directory (comment out whichever you're not currently using)
os.chdir('c:/Users/SchaunW/Dropbox/Code/GitHub/LC_Project/Python/') # Windows
#os.chdir('/Users/schaunwheeler/Dropbox/Code/GitHub/LC_Project/Python/') # OS X

# Scripts
import lc_data

#-----------------------------------------------------------------------------#
# Workflow
#-----------------------------------------------------------------------------#

# Import baseline data

lc = lc_data.LC_Data()

lc.data_import(
    lc_end_file = 'LoanStatsNew.csv',
    lc_monthly_file = 'Pmthist All Loans Version 20130416.csv',
    wd = 'c:/Users/SchaunW/Dropbox/code/lending_club')
#    wd = '/Users/schaunwheeler/Dropbox/code/lending_club')

lc.make_tdms(
    text_fields = ['emp_name', 'desc', 'purpose', 'title'],
    set_min_df = round(lc.dataset.shape[0] * 0.005), set_ngram_range = (1,4),
    count_words = True, count_percents = True, count_currency = True, 
    merge_all = True)