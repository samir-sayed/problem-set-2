'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl as etl
import part2_preprocessing as preprocessing
import part3_logistic_regression as logistic
import part4_decision_tree as decision
import part5_calibration_plot as calibration

# Call functions / instanciate objects from the .py files
def main():
    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.etl_function()

    # PART 2: Call functions/instanciate objects from preprocessing
    preprocessing.preprocessing_function()

    # PART 3: Call functions/instanciate objects from logistic_regression
    logistic.logistic_regression_function()

    # PART 4: Call functions/instanciate objects from decision_tree
    decision.decision_tree_function()

    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration.calibration_function()

if __name__ == "__main__":
    main()
