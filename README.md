<h1 align="center">
	<img width="300" src="https://raw.githubusercontent.com/mindsdb/mindsdb/master/assets/MindsDBColorPurp%403x.png" alt="MindsDB">
	<br>
	<br>
</h1>

# MindsDB Examples
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZoranPandovski/mindsdb-examples/blob/master/)
[![Discourse posts](https://img.shields.io/discourse/posts?server=https%3A%2F%2Fcommunity.mindsdb.com%2F)](https://community.mindsdb.com/)
[![Gitter](https://img.shields.io/gitter/room/mindsdb/community)](https://gitter.im/mindsdb/community)

This repository contains examples of [MindsDB](https://www.mindsdb.com/) usage in predicting different types of data.


## Installation

``
 pip install mindsdb 
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

Inside dataset directory you can find dataset with Test data. e.g benchmarks/home_rentals/dataset/home_rentals_train.csv. You can use this dataset to check the prediction accuracy.

```
cd benchmarks/home_rentals
python3 mindsdb_acc.py
```

## Simple Usage
Lets make our predictions for which we will model the relationship between the three variables and rental price. e.g
```python
Predictor(name='home_rentals').predict(when={'number_of_rooms': 3, 'number_of_bathrooms': 1, 'neighborhood' : 'south_side'})
```
MindsDB will automatically predict a rental price cost value given number_of_rooms, number_of_bathrooms and neighborhood parameters, e.g
> * We are 77% confident the value of "rental_price" lies between 4586 and 4960.

## MindsDB Demo
Check the following tutorial, to learn more about MindsDB end-to-end. 

[![Mindsdb Tutorial](https://img.youtube.com/vi/a49CvkoOdfY/0.jpg)](https://youtu.be/yr7fgqt9cfU) 
