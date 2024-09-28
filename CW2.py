# COMP2611-Artificial Intelligence-Coursework#2 - Descision Trees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os

# STUDENT NAME: Oluwakorede Lasekan
# STUDENT EMAIL: sc22ol@leeds.ac.uk
    
def print_tree_structure(model, header_list):
    tree_rules = export_text(model, feature_names=header_list[:-1])
    print(tree_rules)
    
# Task 1 [10 marks]: Load the data from the CSV file and give back the number of rows
def load_data(file_path, delimiter=','):
    num_rows, data, header_list=None, None, None
    if not os.path.isfile(file_path):
        warnings.warn(f"Task 1: Warning - CSV file '{file_path}' does not exist.")
        return None, None, None
    df = pd.read_csv(file_path, delimiter=delimiter)
    num_rows = len(df)
    header_list = list(df.columns)
    data = np.array(df)
    
    return num_rows, data, header_list

# Task 2[10 marks]: Give back the data by removing the rows with -99 values 
def filter_data(data):
    filtered_data=[None]*1
    # Insert your code here for task 2
    temp = []
    for row in data:
        if not np.any(row == -99):
            temp.append(row)

    filtered_data = temp
    return filtered_data

# Task 3 [10 marks]: Data statistics, return the coefficient of variation for each feature, make sure to remove the rows with nan before doing this. 
def statistics_data(data):
    coefficient_of_variation=None
    data=filter_data(data)
    # Calculate the mean and standard deviation for each feature
    # Insert your code here for task 3
    filtered_data = []
    for row in data:
        if not np.any(np.isnan(row)):
            filtered_data.append(row)
    data = filtered_data
    
    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)
    coefficient_of_variation = std_devs / means
    
    return coefficient_of_variation

# Task 4 [10 marks]: Split the dataset into training (70%) and testing sets (30%), 
# use train_test_split of scikit-learn to make sure that the sampling is stratified, 
# meaning that the ratio between 0 and 1 in the lable column stays the same in train and test groups.
# Also when using train_test_split function from scikit-learn make sure to use "random_state=1" as an argument. 
def split_data(data, test_size=0.3, random_state=1):
    x_train, x_test, y_train, y_test = None, None, None, None
    np.random.seed(1)
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]   
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    return x_train, x_test, y_train, y_test

# Task 5 [10 marks]: Train a decision tree model with cost complexity parameter of 0
def train_decision_tree(x_train, y_train,ccp_alpha=0):
    model=None
    # Insert your code here for task 5
    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    model.fit(x_train,y_train)
    return model

# Task 6 [10 marks]: Make predictions on the testing set 
def make_predictions(model, X_test):
    y_test_predicted=None
    # Insert your code here for task 6
    if model is not None:
        y_test_predicted = model.predict(X_test)
    return y_test_predicted

# Task 7 [10 marks]: Evaluate the model performance by taking test dataset and giving back the accuracy and recall 
def evaluate_model(model, x, y):
    accuracy, recall=None,None
    # Insert your code here for task 7
    if model is not None:
        y_pred = model.predict(x)
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred)
    return accuracy, recall

# Task 8 [10 marks]: Write a function that gives the optimal value for cost complexity parameter
# which leads to simpler model but almost same test accuracy as the unpruned model (+-1% of the unpruned accuracy)
def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    optimal_ccp_alpha=None

    # Insert your code here for task 8
    ccp_alphas = np.linspace(0, 0.1,50)
    min_diff = np.inf
    
    unpruned_model = DecisionTreeClassifier()
    unpruned_model.fit(x_train, y_train)
    unpruned_accuracy = accuracy_score(y_test, unpruned_model.predict(x_test))

    for alpha in ccp_alphas:
        model = DecisionTreeClassifier(ccp_alpha=alpha)
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))

        if abs(accuracy - unpruned_accuracy) <= 0.01 * unpruned_accuracy and abs(accuracy - unpruned_accuracy) < min_diff:
            optimal_ccp_alpha = alpha
            min_diff = abs(accuracy - unpruned_accuracy)

    return optimal_ccp_alpha

# Task 9 [10 marks]: Write a function that gives the depth of a decision tree that it takes as input.
def tree_depths(model):
    depth=None
    # Get the depth of the unpruned tree
    # Insert your code here for task 9
    if model is not None:
        depth = model.tree_.max_depth
    return depth

 # Task 10 [10 marks]: Feature importance 
def important_feature(x_train, y_train,header_list):
    best_feature=None
    # Train decision tree model and increase Cost Complexity Parameter until the depth reaches 1
    # Insert your code here for task 10
    for alpha in np.linspace(0, 0.1, 100):
        model = DecisionTreeClassifier(ccp_alpha=alpha)
        model.fit(x_train, y_train)
        if model.tree_.max_depth == 1:
            feature_index = np.argmax(model.feature_importances_)
            best_feature = header_list[feature_index]
            break
        
    return best_feature

# Example usage (Template Main section):
if __name__ == "__main__":
    # Load data
    file_path = "DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}"); 
    print("-" * 50)


    # Filter data
    data_filtered = filter_data(data)
    #num_rows_filtered=data_filtered.shape[0]
    #print(f"Data is filtered. Number of Rows: {num_rows_filtered}"); 
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)
    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)
    
    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print_tree_structure(model, header_list)
    print("-" * 50)
    
    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)
    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)
    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)
    # Find optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    #print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print(f"optimal_alpha: {optimal_alpha:.4f}")
    print("-" * 50)
    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)
    
    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)
    
    # Feature importance
    important_feature_name = important_feature(x_train, y_train,header_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)
        
# References: 
# Here please provide recognition to any source if you have used or got code snippets from
# Please tell the lines that are relavant to that reference.
# https://scikit-learn.org/stable/index.html for looking at documentation
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# https://stackoverflow.com/questions/54884252/check-the-accuracy-of-decision-tree-classifier-with-python
# https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
# https://stackoverflow.com/questions/54499114/using-sklearn-how-do-i-find-depth-of-a-decision-tree



