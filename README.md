LC_Project
==========

We're going to build this project as a set of modules. Our main workflow will be in `lc_main.py`. 

This is the list of current modules:

## lc_data.py
This module contains functions to change the raw data into a format that is ammenable to analysis. The specific functions 
include:

* `data_import` takes the file names of the end-state and monthly files as well as a working directory. It imports,
  cleans, a merges the files, creating two outcomes variables `return_amount` and `return_percent` for use in subsequent
  analyses.

* `make_tdms` takes any number of pandas series objects and turns them into a dictionary of term-document matrices. The
  function can specify whether to count the total number of words as well as the total occurences of percentages and 
  currency (dollars only) in each document. The `set_min_df` parameter allows you to set the minimum number of documents
  in which a term must appear in order to be included in the term-document matrix. The `set_ngram_range` parameter allows
  you to specifically look for bigrams, trigrams, etc.

* `percent_to_float` is a helper function that is called within `data_import`. It converts percentages represented as 
  strings into floats.

* `clean_text` is a helper function called within `make_tdms`. It removes non-alphabetical characters, stems words (if
  desired), and in the case of the `desc` column in the main data set, it removes all updates that were made to the 
  field after the application was first posted.
