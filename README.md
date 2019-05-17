# BrainAgePredictor

## Installation
This work uses Python 3.6.8 (64-bit) with packages numpy, sklearn, pandas, scipy, csv, and matplotlib. Ensure python is downloaded, then install the packages with

```
$ pip install PACKAGE_NAME
```

from the command line. Ensure the Python executable has been added to your system's PATH variable.

## Running the code
Run the code from the command line with

```
$ python train.py
```

Alternatively, the code can be ran within an IDE, such as Visual Studio Code.

## train.py
This script houses all of the statistical analysis and model creation in a sequential script. Data is loaded from desired CSVs, scaled, a model is fit, then scores or other outputs are printed. Currently two graphs are created as well using matplotlib. See commends within the script for adjusting the results.

## utility_functions.py
A handful of functions found useful while developing. Some are old and unused, but the main CSV loader function resides here.

## All other excel and powerpoint files
All other files house useful information saved during analysis of the models on subsets of data. 
PredictedAgesFromModels is an up-to-date excel doc with R^2 values to determine which models work best on which datasets. 
PredictedAges is an older excel doc that is reused to make charts to add into the powerpoint presentations.
All CSV files were used for training at one point or another.
