# -*- coding: utf-8 -*-
"""Some helper functions for project 2."""
###### Basics

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import statsmodels.api as sm

##### Sklearn
## metrics

from sklearn.metrics import f1_score,accuracy_score,precision_recall_curve,roc_auc_score,roc_curve, auc

## models

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.feature_selection import RFE

#### yellowbrick

from yellowbrick.classifier.rocauc import roc_auc
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ClassPredictionError, ROCAUC,PrecisionRecallCurve
from yellowbrick.features import PCA as PCA_3D
from yellowbrick.features import Rank2D
from yellowbrick.target import ClassBalance
from yellowbrick.features.radviz import radviz

########################################### FEATURE - ENGINEERING ############################################################


def delete_minus1(X): 
    """ Delete columns and rows which contain unknowns (-1)
    we first start by removing columns and then the rows

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - Xp : A matrix with no missing values
    
    """
    Xp = X.copy()
    for column in Xp:

        if sum(Xp[column]==-1) >= 0.5 * len(Xp[column]):
            Xp.drop(column, axis=1, inplace = True)
            continue

        index_names = Xp[(Xp[column] == -1)].index;
        Xp.drop(index_names, inplace = True)
    
    return Xp


def make_ones_indicator_column(data_frame, name_of_column_target, inplace=False):
    """ For a desired column we add an indicator column full of '1's, of the same name as the target column but
    with suffix '_indicator'.
    If the indicator column already exists: function does nothing.
    
    Parameters
    ----------
    data_frame: A data frame
    
    name_of_column_target: Column which will be given an indicator column
    
    inplace = State whether the change must be a copy or an inplace operation (Default = False)

    Returns
    -------
    
    - Returns the indicator column filled with ones and zeros
    
    """
    if name_of_column_target + '_indicator' in data_frame.columns:
        if not inplace:
            return data_frame.copy()
    else:
        if inplace:
            data_frame[name_of_column_target + '_indicator'] = np.ones(data_frame[name_of_column_target].size)
        else :
            df_temp = data_frame.copy()
            df_temp[name_of_column_target + '_indicator'] = np.ones(df_temp[name_of_column_target].size)
            return df_temp


def put_zero_in_indicator_column(data_frame, name_of_column_target, target_value, inplace=False):
    """ Finds in the indicator column the lines where the target column has target value, and puts 0 there
    

    Parameters
    ----------
    data_frame: A data frame
    
    name_of_column_target: Column which will be given an indicator column
    
    target_value: value that will be change to zero each time it is encountered

    inplace = State whether the change must be a copy or an inplace operation (Default = False)
    Returns
    -------
    
    - A data frame where zeros will be added to the indicator column
    
    """
    if inplace:
        data_frame.loc[data_frame[name_of_column_target] == target_value, name_of_column_target + '_indicator'] = 0
    else :
        df_temp = data_frame.copy()
        df_temp.loc[df_temp[name_of_column_target] == target_value, name_of_column_target + '_indicator'] = 0
        return df_temp


def make_indicator_for_bad_data(data_frame, name_of_column_target, target_value, inplace=False):
    """ Adds a column to the right of data_frame, with name of target + _indicator,
    with 0 on same line as the target column has target value
    (If the indicator column already exists, it will work on it instead of creating another)

    Parameters
    ----------
    data_frame: a data frame
    
    name_of_column_target: The name of the column to make an indicator for
    
    target_value: The value to be associated with 0 in the indicator column
    
    inplace = State whether the change must be a copy or an inplace operation (Default = False)

    Returns
    -------
    
    - A data frame with an extra indicator column for the target column, highlighting the target value
    
    """
    if inplace:
        make_ones_indicator_column(data_frame, name_of_column_target, inplace)
        put_zero_in_indicator_column(data_frame, name_of_column_target, target_value, inplace)
    else :
        df_temp = make_ones_indicator_column(data_frame, name_of_column_target, inplace=False)
        put_zero_in_indicator_column(df_temp, name_of_column_target, target_value, inplace=True)
        return df_temp


def make_indicators(data_frame, list_of_column_target, list_of_target_values, inplace=False):
    """ Iteratively calls make_indicator_for_bad_data for on the data frame,
    for each element of the lists of columns and target values.
    (Putting the same column name in the list several times allows to hunt multiple target values in said column)

    Parameters
    ----------
    data_frame: a data frame
    
    list_of_column_target: a list of the names of the columns on which to make indicators
    
    list_of_target_values: a list of target values for the corresponding target column in list_of_target_columns
    
    inplace = State whether the change must be a copy or an inplace operation (Default = False)

    Returns
    -------
    
    - The data frame with all the indicator columns appended.
    
    """
    if not inplace:
        df_temp = data_frame.copy()
    else:
        df_temp = data_frame
    for i, col in enumerate(list_of_column_target):
        make_indicator_for_bad_data(df_temp, col, list_of_target_values[i], inplace=True)
    if not inplace:
        return df_temp
    

def extract_certain_dataset(data_frame, name_of_target_column, target_value):
    """ Extracts a subset of the dataset, where all the rows of the target column have target value

    Parameters
    ----------
    data_frame: a data frame
    
    name_of_target_column: the name of the column to target
    
    target_value: the value to extract the subset around

    Returns
    -------
    
    - a data frame (subset of data_frame) with all rows with target value in target column
    
    """
    df = data_frame[data_frame[name_of_target_column] == target_value].copy()
    return df


def make_list_by_value(data_frame, name_of_target_column, name_of_reference_column):
    """  Will create a list of data frames all subset disjoint of data_frame, for all values the target column can take.
    (It will also set the reference column as index for all the data frames)

    Parameters
    ----------
    data_frame: a data frame
    
    name_of_target_column: the name of the column whose entries will determine the separation
    
    name_of_reference_column: the name of the column to set as index

    Returns
    -------
    
    - a list of data frames.
    
    """
    list_of_df = []
    list_of_target_values = data_frame[name_of_target_column].value_counts().index.sort_values()
    for i, value in enumerate(list_of_target_values):
        list_of_df.append(extract_certain_dataset(data_frame, name_of_target_column, value).set_index(name_of_reference_column))
    return list_of_df


def rearange_horizontally(data_frame, name_of_target_column, name_of_reference_column):
    """ First separates the original dataset into subgroups corresponding to values on the target column,
    and then puts them back together, but horizontally instead of vertically. Aligned around the reference column.
    This function is used to create the data frame that separates the days
    If data_frame is of shape (n x m) and the target column takes k different values (assuming they are balanced),
    then the output will be of size (n/k x km)


    Parameters
    ----------
    
    data_frame: a data frame
    
    name_of_target_column: the name of the column around which the data frame will be rearranged
    
    name_of_reference_column: the name of the column to set as index

    Returns
    -------
    
    - a data frame. It contains exactly the same data as data_frame,
        but rearranged horizontally following the values of the target column
    
    """
    list_of_df = make_list_by_value(data_frame, name_of_target_column, name_of_reference_column)
    return pd.concat(list_of_df, axis=1, sort=False)


def dissassemble(data_frame, name_of_column_target):
    """ makes a list of data frames subsets disjoint of data_frame. Almost identical to make_list_by_value.
    The only difference with make_list_by_value is that there aren't any column set to index

    Parameters
    ----------
    
    data_frame: a data frame
    
    name_of_column_target: name of the column around which the separation is done

    Returns
    -------
    
    - a list of data frames
    
    """
    out_list=[]
    for i, value in enumerate(data_frame[name_of_column_target].value_counts(sort=False).index):
        df = data_frame[data_frame[name_of_column_target] == value].copy()
        out_list.append(df)
    return out_list


def fine_dissassembly(data_frame, name_first_column_target, name_second_column_target):
    """ Calls dissassemble twice to output a matrix of subsets disjoint of data_frame.

    Parameters
    ----------
    data_frame: a data frame
    
    name_first_column_target: the name of the column around which the first separation will be made
    
    name_second_column_target: the name of the column for the second separation

    Returns
    -------
    
    - a list of list of data frames.
    
    """
    
    out_list = dissassemble(data_frame, name_first_column_target)
    for i, subdf in enumerate(out_list):
        out_list[i] = dissassemble(subdf, name_second_column_target)
    return out_list


def reassemble(list_of_df):
    """ Does the opposit of dissassemble.
    Makes a single data frame out of a list of data frames by concatenating them
    
    Parameters
    ----------
    list_of_df: a list of data frames

    Returns
    -------
    
    - a data frame
    
    """
    return pd.concat(list_of_df)


def big_reassembly(list_of_list_of_df):
    """ Does the opposite of fine_dissassembly.
    Creates a single data frame out of a matrix of smaller data frames

    Parameters
    ----------
    list_of_list_of_df: a list of lists of data frames

    Returns
    -------
    
    - a single data frame. The concatenation of all of the data frames
    
    """
    for i, sublist in enumerate(list_of_list_of_df):
        list_of_list_of_df[i] = reassemble(sublist)
    return reassemble(list_of_list_of_df)
   

def subtract_list(list1, list2):
    """ Substracts two lists and return the difference.
    Creates a list with all the elements of list1 except those that are also in list2

    Parameters
    ----------
    list1: A list
    
    list2: A list

    Returns
    -------
    
    - A list. The difference between the two lists
    
    """
    list3= []
    for i in list1:
        if i not in list2:
            list3.append(i)
    return list3


def transform_into_horizontal_df(data_frame, 
                                 reference_column='msfid', 
                                 current_time_column='datclin', 
                                 first_day_column='first_date', 
                                 no_need_duplicate_columns=['sex', 'dt', 'time_stayed', 'outcome', 'datsym', 'age'],
                                 columns_to_readd_at_end=['sex', 'age', 'outcome']):
    """ Function to rearrange the data frame into an horizontal version with most of the columns repeated,
    except those blacklisted.
    The function is very situational.
    Recommended use : df = transform_into_horizontal_df(data_frame) (when data_frame has only one observation per day)

    Parameters
    ----------
    data_frame: Input data frame
    
    reference_column: (Default = 'msfid')
    
    current_time_column: (Default = 'datclin')
    
    first_day_column: (Default = 'first_date')
    
    no_need_duplicate_columns: Columns that should not be duplicated when assembling the dataframes (Default = ['sex', 'dt', 'time_stayed', 'outcome', 'datsym', 'age'])
    
    columns_to_readd_at_end: Columns that will be added at the end of the dataframe (Default = ['sex', 'age', 'outcome'])

    Returns
    -------
    
    - Returns the original data frame concatenated horizontally by "time_elapsed"
    
    """
    rest = subtract_list(data_frame.columns, no_need_duplicate_columns)
    df_to_rearange = data_frame[rest].copy()    # Finds the columns that have information that depends on the day
    df_to_rearange['time_elapsed'] = df_to_rearange[current_time_column] - df_to_rearange[first_day_column]
    df_rearanged = rearange_horizontally(df_to_rearange, 'time_elapsed', reference_column)  # rearranges around the day
    df_rearanged = df_rearanged.reset_index()
    df_rearanged = df_rearanged.rename(index=str, columns={'index':'msfid'})    # somehow the name of the index is lost
    df_tail = data_frame[[reference_column] + columns_to_readd_at_end].copy()   # get the columns to re-add
    df_tail_shrunk = df_tail.groupby(reference_column).nth(0)                   # transform them to the right shape
    df_rearanged_with_end = pd.merge(df_rearanged.set_index(reference_column), df_tail_shrunk, left_index=True, right_index=True, how='inner')
    return df_rearanged_with_end.reset_index()


########################################### DATA - VISUALIZATION ##############################################################


def Corr_vision(X):
    """ Correlation visualization according to Pearson

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """


    fig, ax = plt.subplots(figsize=(20,20))
    visualizer = Rank2D(algorithm="pearson")
    visualizer.fit_transform(X)
    #visualizer.show('corr_matrix') // to output png
    plt.show()


def Imbalance(y):
    """ Imabalance between the labels

    Parameters
    ----------
    y: vector of labels

    Returns
    -------
    
    - A plot with the class imbalances for Ebola positive or negative
    
    """
    # Instantiate the visualizer
    visualizer = ClassBalance(labels=['Ebola negative', 'Ebola positive'])

    visualizer.fit(y)                        # Fit the data to the visualizer
    #visualizer.show('class_balance')        # Finalize and render the figure
    plt.show()


def Imbalance_out(y):
    """ Imabalance between the labels

    Parameters
    ----------
    y: vector of labels

    Returns
    -------
    
    - A plot with the class imbalances for the outcome
    
    """
    # Instantiate the visualizer
    visualizer = ClassBalance(labels=[ 'Survival','Death'])

    visualizer.fit(y)                        # Fit the data to the visualizer
    #visualizer.show('class_balance')        # Finalize and render the figure
    plt.show()


def Rad_vision(X,y):
    """ Radial distributions of cases around the systems, a method to detect separability between classes

    Parameters
    ----------
    X: matrix of features
    
    y: vector of labels, for diagnosis

    Returns
    -------
    
    - A radial plot, with the labels and the features at the circumference
    
    """

    fig, ax = plt.subplots(figsize=(20,10))
    radviz(X, y.values, classes = ['Ebola negative', 'Ebola positive'])
    plt.show()


def Rad_vision_out(X,y):
    """ Radial distributions of cases around the systems, a method to detect separability between classes

    Parameters
    ----------
    X: matrix of features
    
    y: vector of labels, for prognosis

    Returns
    -------
    
    - A radial plot, with the labels and the features at the circumference
    
    """
    fig, ax = plt.subplots(figsize=(20,10))
    radviz(X, y.values, classes = ['Survival', 'Death'])
    plt.show()
    

# Different functions due to different labels
def score_model(X_train, y_train, X_test, y_test, model,  **kwargs):
    """ A function that returns the different metrics of accuracy, confusion matrix and other model reports depending on the type of model that is asked.
    
    This function is for diagnosis, please use score_model_outcome for prognosis

    Parameters
    ----------
    X_train: matrix of training features
    
    y_train: vector of training labels
    
    X_test: matrix of test features
    
    y_test: vector of test labels

    Returns
    -------
    
    - Accuracy, F1 score and ROC_AUC for the train and test set
    
    - Confusion matrix
    
    - ClassificationReport
    
    - PrecisionRecallCurve
    
    - ClassPredictionError
    
    """

    # Train the model
    model.fit(X_train, y_train, **kwargs)
    
    # Predict on the train set
    prediction_train = model.predict(X_train)
    
    # Compute metrics for the train set
    accuracy_train = accuracy_score(y_train, prediction_train)
    
    # False Positive Rate, True Positive Rate, Threshold
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, prediction_train)
    auc_train = auc(fpr_train, tpr_train)
    
    f1_score_train = f1_score(y_train, prediction_train)

    # Predict on the test set
    prediction_test = model.predict(X_test)
    
    accuracy_test = accuracy_score(y_test, prediction_test)

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, prediction_test)
    auc_test = auc(fpr_test, tpr_test)
    
    f1_score_test = f1_score(y_test, prediction_test)
    
    print("{}:".format(model.__class__.__name__))
    # Compute and return F1 (harmonic mean of precision and recall)
    print("On training we get an Accuracy {}, an AUC {} and F1 score {} ".format(accuracy_train, auc_train, f1_score_train ) )
    
    print("For test we get an Accuracy {}, an AUC {} and F1 score {}".format(accuracy_test, auc_test, f1_score_test) )
    
    fig, axes = plt.subplots(3, 2, figsize = (20,20))

    visualgrid = [
        ConfusionMatrix(model, ax=axes[0][0], classes=['Ebola Negative', 'Ebola Positive'], cmap="YlGnBu"),
        ClassificationReport(model, ax=axes[0][1], classes=['Ebola Negative', 'Ebola Positive'],cmap="YlGn",),
        PrecisionRecallCurve(model, ax=axes[1][0]),
        ClassPredictionError(model, classes=['Ebola Negative', 'Ebola Positive'], ax=axes[1][1]),
    ]

    for viz in visualgrid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
    
    try:
        roc_auc(model, X_train, y_train, X_test=X_test, y_test=y_test, classes=['Ebola Negative', 'Ebola Positive'], ax=axes[2][0])
    except:
        print('Can plot ROC curve for this model')
    
    try:
        viz = FeatureImportances(model,ax=axes[2][1], stack=True, relative=False)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
    except:
        print('Don\'t have feature importance')
        
    plt.show()
    print('\n')


def score_model_outcome(X_train, y_train, X_test, y_test, model,  **kwargs):
    """ A function that returns the different metrics of accuracy, confusion matrix and other model reports depending on the type of model that is asked.
    
    This function is for prognosis

    Parameters
    ----------
    X_train: matrix of training features
    
    y_train: vector of training labels
    
    X_test: matrix of test features
    
    y_test: vector of test labels

    Returns
    -------
    
    - Accuracy, F1 score and ROC_AUC for the train and test set
    
    - Confusion matrix
    
    - ClassificationReport
    
    - PrecisionRecallCurve
    
    - ClassPredictionError
    
    """
    
    # Train the model
    model.fit(X_train, y_train, **kwargs)
    
    # Predict on the train set
    prediction_train = model.predict(X_train)
    
    # Compute metrics for the train set
    accuracy_train = accuracy_score(y_train, prediction_train)
    
    # False Positive Rate, True Positive Rate, Threshold
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, prediction_train)
    auc_train = auc(fpr_train, tpr_train)
    
    f1_score_train = f1_score(y_train, prediction_train)

    # Predict on the test set
    prediction_test = model.predict(X_test)
    
    accuracy_test = accuracy_score(y_test, prediction_test)

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, prediction_test)
    auc_test = auc(fpr_test, tpr_test)
    
    f1_score_test = f1_score(y_test, prediction_test)
    
    print("{}:".format(model.__class__.__name__))
    # Compute and return F1 (harmonic mean of precision and recall)
    print("On training we get an Accuracy {}, an AUC {} and F1 score {} ".format(accuracy_train, auc_train, f1_score_train ) )
    
    print("For test we get an Accuracy {}, an AUC {} and F1 score {}".format(accuracy_test, auc_test, f1_score_test) )
    
    fig, axes = plt.subplots(3, 2, figsize = (20,20))

    visualgrid = [
        ConfusionMatrix(model, ax=axes[0][0], classes=['Death', 'Survival'], cmap="YlGnBu"),
        ClassificationReport(model, ax=axes[0][1], classes=['Death', 'Survival'],cmap="YlGn",),
        PrecisionRecallCurve(model, ax=axes[1][0]),
        ClassPredictionError(model, classes=['Death', 'Survival'], ax=axes[1][1]),
    ]

    for viz in visualgrid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
    
    try:
        roc_auc(model, X_train, y_train, X_test=X_test, y_test=y_test, classes=['Death', 'Survival'], ax=axes[2][0])
    except:
        print('Can plot ROC curve for this model')
    
    try:
        viz = FeatureImportances(model,ax=axes[2][1], stack=True, relative=False)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
    except:
        print('Don\'t have feature importance')
        
    plt.show()
    print('\n')


# Different functions due to different labels
def PCA_vision_3D(X,y):
    """ Visualizes 3D PCA for diagnosis


    Parameters
    ----------
    X: matrix of features
    
    y: target labels for diagnosis

    Returns
    -------
    
    - 3D PCA plot 
    
    """
    visualizer = PCA_3D(scale=True, projection=3, classes=['Ebola negative', 'Ebola positive'])
    visualizer.fit_transform(X, y)
    #visualizer.show()
    plt.show()


def PCA_vision_3D_out(X,y):
    """ Visualizes 3D PCA for prognosis


    Parameters
    ----------
    X: matrix of features
    
    y: target labels for prognosis

    Returns
    -------
    
    - 3D PCA plot for prognosis
    
    """
    
    visualizer = PCA_3D(scale=True, projection=3, classes=['Survival', 'Death'])
    visualizer.fit_transform(X, y)
    #visualizer.show()
    plt.show()




