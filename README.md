# Data-mining-NYC-taxi-trip-duration-prediction
## Introduction
In this data mining project, the goal of the problem is to predict New York City taxi trip duration, which is a supervised-learning regression problem. I will approach the problem in several steps, namely **exploratory data analysis, preprocessing phase, feature engineerings and xgboost modeling**.

This project makes references to [Kaggle experts sharing kernel](https://github.com/mxbi/mlnd-capstone).

## Requirements
```
pip install -r requirements.txt
```

## Dataset
The dataset came from the Kaggle competition: https://www.kaggle.com/c/nyc-taxi-trip-duration/data

## Usage
To run the code, following steps are needed.
1. ``python preprocessing.py``
2. ``python feature_engineering.py``
3. ``python model.py`` (initial xgboost model)
4. ``python parameters_optim.py`` (tunning xgboost model parameters with Bayesian optimization)
5. ``python model_optimization.py`` 
