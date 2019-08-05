# Mindsdb Examples

This repository contains examples of mindsdb usage in predicting different types of data.


## Installation
``
 pip3 install mindsdb --user
 
``

## Train 

In each directory there are different types of datasets avaiable.

``
cd wine_quality
python3 train.py
``

## Predict

Inside dataset directory you can find dataset with Test data. e.g wine_quality/dataset/WineQualityTest.csv. You can use values from this dataset to check the accuracy for prediction.

``
cd wine_quality
python3 predict.py
``
