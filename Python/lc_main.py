# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Setup
#-----------------------------------------------------------------------------#

# All dependencies for alll scripts (for reference - we'll delete this later)
import os
#import numpy as np
#import pandas as pd
#import re


# Set working directory (comment out whichever you're not currently using)
os.chdir('c:/Users/SchaunW/Dropbox/GitHub/LC_Project/Python/') # Windows
#os.chdir('~/Dropbox/GitHub/LC_Project/Python/') # OS X

# Scripts
import lc_data

#-----------------------------------------------------------------------------#
# Workflow
#-----------------------------------------------------------------------------#

# Import baseline data
lc = lc_data.lc_data_import(
    lc_end_file = 'LoanStatsNew.csv',
    lc_monthly_file = 'Pmthist All Loans Version 20130416.csv',
    wd = 'c:/Users/SchaunW/Dropbox/code/lending_club')