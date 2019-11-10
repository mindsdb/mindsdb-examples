<h1 align="center">
	<img width="300" src="https://raw.githubusercontent.com/mindsdb/mindsdb/master/assets/logo_gh.png" alt="MindsDB">
	<br>
	<br>
</h1>

# Mindsdb Examples [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZoranPandovski/mindsdb-examples/blob/master/)


This repository contains examples of [MindsDB](https://www.mindsdb.com/) usage in predicting different types of data.


## Installation

``
 pip3 install mindsdb --user
``
or
``
pip install -r requirements.txt
``

## Train 

In each directory there are different types of datasets avaiable.

```
cd home_rentals

python3 train.py
```

## Predict

Inside dataset directory you can find dataset with Test data. e.g wine_quality/dataset/WineQualityTest.csv. You can use values from this dataset to check the accuracy for prediction.

```
cd home_rentals


python3 predict.py
```

## Simple Usage
Lets make our predictions for which we will model the relationship between the three variables and rental price. e.g
```python
Predictor(name='home_rentals').predict(when={'number_of_rooms': 3, 'number_of_bathrooms': 1, 'neighborhood' : 'south_side'})
```
Mindsdb will automatically predict a rental price cost value given number_of_rooms, number_of_bathrooms and neighborhood parameters, e.g
> * We are 37% confident the value of "rental_price" lies between 4586 and 4960.
