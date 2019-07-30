# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:19:58 2019

@author: tatha
"""

import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier 

from sklearn.model_selection import GridSearchCV



data=pd.read_csv("pima-indians-diabetes.csv")

print(data.describe())
print(data.keys())

X_data=data.drop(["Class","Group"],axis=1)
y_data=data["Class"]


variable_params = {'max_depth':[2,4,6,10], 'n_estimators':[5, 10, 20, 25], 'learning_rate':np.linspace(1e-16, 1 , 3)}
static_params = {'objective':'multi:softmax','num_class':4, 'silent':1}

bst_grid = GridSearchCV (
        estimator = XGBClassifier(**static_params),
        param_grid = variable_params,
        scoring = "accuracy"
        )


bst_grid.fit(X_data, y_data)

print("Best Accuracy:{}".format(bst_grid.best_score_))
for key,value in bst_grid.best_params_.items():
    print("{}:{}".format(key,value))