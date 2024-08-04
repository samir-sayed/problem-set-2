'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=5):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()
df_arrests_train = pd.read_csv('data/df_arrests_train.csv')
df_arrests_test = pd.read_csv('data/df_arrests_test.csv')


features = ['current_charge_felony', 'num_fel_arrests_last_year']

param_grid = {'C': [0.01, 0.1, 1, 10, 100]} #logistic model
lr_model = LogisticRegression()
gs_cv = GridSearchCV(lr_model, param_grid, cv=5)
gs_cv.fit(df_arrests_train[features], df_arrests_train['y'])

lr_probs = gs_cv.predict_proba(df_arrests_test[features])[:, 1]
calibration_plot(df_arrests_test['y'], lr_probs, n_bins=5)

param_grid_dt = {'max_depth': [3, 5, 7, 10, 15]}
dt_model = DTC()
gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
gs_cv_dt.fit(df_arrests_train[features], df_arrests_train['y'])


dt_probs = gs_cv_dt.predict_proba(df_arrests_test[features])[:, 1]
calibration_plot(df_arrests_test['y'], dt_probs, n_bins=5)


print("Which model is more calibrated?: The Decision Tree is more calibrated")