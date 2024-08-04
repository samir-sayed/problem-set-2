'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

def decision_tree_function():
    df_arrests_train = pd.read_csv('data/df_arrests_train.csv') # load csv
    df_arrests_test = pd.read_csv('data/df_arrests_test.csv')

    param_grid_dt = {'max_depth': [3, 5, 7, 10, 15]} # create parameter grid

    dt_model = DTC() # initialize tree model

    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5) # initialize GridSearchCV

    gs_cv_dt.fit(df_arrests_train[['current_charge_felony', 'num_fel_arrests_last_year']], df_arrests_train['y']) # run model

    optimal_max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"The optimal value for max_depth: {optimal_max_depth}")

    if optimal_max_depth == min(param_grid_dt['max_depth']):
        regularization = "least"
    elif optimal_max_depth == max(param_grid_dt['max_depth']):
        regularization = "most"
    else:
        regularization = "middle"

    print(f"What was the optimal value for max_depth? Did it have the most or least regularization? Or in the middle? It has the {regularization} regularization")

    df_arrests_test['pred_dt'] = gs_cv_dt.predict(df_arrests_test[['current_charge_felony', 'num_fel_arrests_last_year']])

    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)