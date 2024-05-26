# Databricks notebook source
## Some Useful User-Defined-Functions
import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
from termcolor import colored
from termcolor import cprint
import missingno as msno 
###############################################################################

def missing_values(df):
    missing_number = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values[missing_values['Missing_Number']>0]

###############################################################################

def first_looking(df):
    print(colored("Shape:", 'yellow', attrs=['bold']), df.shape,'\n', 
          colored('*'*100, 'red', attrs=['bold']),
          colored("\nInfo:\n",'yellow', attrs=['bold']), sep='')
    print(df.info(), '\n', 
          colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("Number of Uniques:\n", 'yellow', attrs=['bold']), df.nunique(),'\n',
          colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("Missing Values:\n", 'yellow', attrs=['bold']), missing_values(df),'\n', 
          colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("All Columns:", 'yellow', attrs=['bold']), *list(df.columns), sep='\n- ') 
    print(colored('*'*100, 'red', attrs=['bold']), sep='')

    df.columns= df.columns.str.lower().str.replace('&', '_').str.replace(' ', '_').str.replace('-', '_')
    
    print(colored("Columns after rename:", 'yellow', attrs=['bold']), *list(df.columns), sep='\n- ')
    print(colored('*'*100, 'red', attrs=['bold']), sep='')
    
###############################################################################

## To view summary information about the columns
def summary(df, column):
    print(colored("Column: ",'yellow', attrs=['bold']), column)
    print(colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("Missing values: ", 'yellow', attrs=['bold']), df[column].isnull().sum())
    print(colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("Missing values(%): ", 'yellow', attrs=['bold']), round(df[column].isnull().sum()/df.shape[0]*100, 2))
    print(colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("Unique values: ", 'yellow', attrs=['bold']), df[column].nunique())
    print(colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("Value counts: \n", 'yellow', attrs=['bold']), pd.DataFrame(df[column].value_counts(dropna = False))\
                                                                                        .rename(columns={column:"count"}).rename_axis(column)\
                                                                                        .assign(percentage=df[column].value_counts(dropna=False, normalize=True)*100), sep='')
    print(colored('*'*100, 'red', attrs=['bold']), sep='')
    
###############################################################################
                    
def multicolinearity_control(df, collimit=0.9):                    
    df_temp = df.corr()
    
    feature =[]
    collinear= []
    for col in df_temp.columns:
        min_corr_value = collimit
        for i in df_temp.index:
            if abs(df_temp[col][i]) > min_corr_value and abs(df_temp[col][i]) < 1:
                feature.append(col)
                collinear.append(i)
                print(f"multicolinearity alert over {min_corr_value} in between {col} - {i}")
    return feature, collinear
    if feature == []:
       print("There is NO multicollinearity problem.")   
                    
###############################################################################

def color(corr_val):
    if abs(corr_val) > 0.8 and abs(corr_val) < 0.9:
        color = 'orange'
    elif abs(corr_val) >= 0.9 and abs(corr_val) <=0.999999:
        color = 'red'
    elif abs(corr_val) == 1:
        color = "white"
    else:
        color = 'black'
    return f'color: {color}'                  
                    
###############################################################################

def duplicate_values(df):
    print(colored("Duplicate check...", 'yellow', attrs=['bold']), sep='')
    duplicate_values = df.duplicated(subset=None, keep='first').sum()
    if duplicate_values > 0:
        df.drop_duplicates(keep='first', inplace=True)
        print(duplicate_values, colored(" Duplicates were dropped!"),'\n',
              colored('*'*100, 'red', attrs=['bold']), sep='')
    else:
        print(colored("There are no duplicates"),'\n',
              colored('*'*100, 'red', attrs=['bold']), sep='')     

###############################################################################
        
def drop_columns(df, drop_columns):
    if drop_columns !=[]:
        df.drop(drop_columns, axis=1, inplace=True)
        print(drop_columns, 'were dropped')
    else:
        print(colored('Missing value control...', 'yellow', attrs=['bold']),'\n',
              colored('If there is a missing value above the limit you have given, the relevant columns are dropped and an information is given.'), sep='')

###############################################################################

def drop_null(df, limit): 
    for i in df.isnull().sum().index:
        if (df.isnull().sum()[i]/df.shape[0]*100)>limit:
            print(df.isnull().sum()[i], 'percent of', i ,'were null and dropped')
            df.drop(i, axis=1, inplace=True)
    print(colored('Last shape after missing value control:', 'yellow', attrs=['bold']), df.shape, '\n', 
          colored('*'*100, 'red', attrs=['bold']), sep='')

############################################################################### 

def shape(df, X, y, X_train, y_train, X_test, y_test, y_pred):
    print('df.shape:', df.shape)
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)
    print('X_test.shape:', X_test.shape)
    print('y_test.shape:', y_test.shape)
    try:
        print('y_pred.shape:', y_pred.shape)
    except:
        print()

###############################################################################

def shape_control(df, X_train, y_train, X_test, y_test):
    print('df.shape:', df.shape)
    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)
    print('X_test.shape:', X_test.shape)
    print('y_test.shape:', y_test.shape)

############################################################################### 

## Show values in bar graphic
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        
############################################################################### 

## Future Selection
def feature_importances(model, X):
    df_fi = pd.DataFrame(index=X.columns, 
                         data=model.feature_importances_, 
                         columns=["Feature Importance"]).sort_values("Feature Importance")

    return df_fi.sort_values(by="Feature Importance", ascending=False).T

def feature_importances_bar(model, X):
    df_fi = pd.DataFrame(index=X.columns, 
                         data=model.feature_importances_, 
                         columns=["Feature Importance"]).sort_values("Feature Importance")

    sns.barplot(data = df_fi, 
                x = df_fi.index, 
                y = 'Feature Importance', 
                order=df_fi.sort_values('Feature Importance', ascending=False).reset_index()['index'])

###############################################################################

def show_distribution_density(col):
    from matplotlib import pyplot as plt

    min_val = col.min()
    max_val = col.max()
    mean_val = col.mean()
    med_val = col.median()
    mod_val = col.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                              mean_val,
                                                                                              med_val,
                                                                                              mod_val,
                                                                                              max_val))

    fig = plt.figure(figsize = (15,10))
    
    plt.subplot(3, 1, 1)
    plt.hist(col)
    plt.ylabel('Frequency', fontsize=10)
    plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2, label='Minimum')
    plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2, label='Mean')
    plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2, label='Median')
    plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2, label='Mode')
    plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2, label='Maximum')
    plt.legend(loc='upper right')
    
    plt.subplot(3, 1, 2) 
    plt.boxplot(col, vert=False)
    
    plt.subplot(3, 1, 3)
    col.plot.density()
    plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2, label='Minimum')
    plt.axvline(x=col.mean(), color = 'cyan', linestyle='dashed', linewidth = 2, label = 'Mean')
    plt.axvline(x=col.median(), color = 'red', linestyle='dashed', linewidth = 2, label = 'Median')
    plt.axvline(x=col.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2, label = 'Mode')
    plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2, label='Maximum')
    plt.legend(loc='upper right')

    fig.suptitle("Distribution", fontsize=15)
    
    plt.show();

###############################################################################

def show_density(col):

    min_val = col.min()
    max_val = col.max()
    mean_val = col.mean()
    med_val = col.median()
    mod_val = col.mode()[0]

    fig = plt.figure(figsize=(15,5))

    col.plot.density()

    plt.title('Data Density', fontsize=15)

    plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2, label='Minimum')
    plt.axvline(x=col.mean(), color = 'cyan', linestyle='dashed', linewidth = 2, label = 'Mean')
    plt.axvline(x=col.median(), color = 'red', linestyle='dashed', linewidth = 2, label = 'Median')
    plt.axvline(x=col.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2, label = 'Mode')
    plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2, label='Maximum')
    plt.legend(loc='upper right')
    plt.show(); 

###############################################################################
# Create a function that we can re-use
def show_distribution(var_data, suptitle):
    from matplotlib import pyplot as plt
    
    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    # print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                             # mean_val,
                                                                                             # med_val,
                                                                                             # mod_val,
                                                                                             # max_val))

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10, 4))

    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # Add a title to the Figure
    fig.suptitle(suptitle)

    # Show the figure
    fig.show();
       
###############################################################################
## Handling and Dealing with Outliers

from scipy import stats

'''This function detects the best z-score for outlier detection in the specified column.'''

def outlier_zscore(df, col, min_z=1, max_z = 5, step = 0.05, print_list = False):
    z_scores = stats.zscore(df[col].dropna())
    threshold_list = []
    
    for threshold in np.arange(min_z, max_z, step):
        threshold_list.append((threshold, len(np.where(z_scores > threshold)[0])))
    
    df_outlier = pd.DataFrame(threshold_list, columns = ['threshold', 'outlier_count'])
    df_outlier['pct'] = (df_outlier.outlier_count - df_outlier.outlier_count.shift(-1))/df_outlier.outlier_count*100
    df_outlier['pct'] = df_outlier['pct'].apply(lambda x : x-100 if x == 100 else x)
    best_treshold = round(df_outlier.iloc[df_outlier.pct.argmax(), 0],2)
    IQR_coef = round((best_treshold - 0.675) / 1.35, 2)
    outlier_limit = int(df[col].dropna().mean() + (df[col].dropna().std()) * df_outlier.iloc[df_outlier.pct.argmax(), 0])
    num_outlier = df_outlier.iloc[df_outlier.pct.argmax(), 1]
    percentile_threshold = stats.percentileofscore(df[col].dropna(), outlier_limit)
    plt.plot(df_outlier.threshold, df_outlier.outlier_count)
    plt.vlines(best_treshold, 0, df_outlier.outlier_count.max(), colors="r", ls = ":")
    plt.annotate("Zscore : {}\nIQR_coef : {}\nValue : {}\nNum_outlier : {}\nPercentile : {}".format(best_treshold,
                                                                          IQR_coef,
                                                                          outlier_limit,
                                                                          num_outlier,     
                                                                          (np.round(percentile_threshold, 3), 
                                                                           np.round(100-percentile_threshold, 3))),
                                                                          (best_treshold, df_outlier.outlier_count.max()/2))
    plt.show()
    if print_list:
        print(df_outlier)
    return (plt, df_outlier, best_treshold, IQR_coef, outlier_limit, num_outlier, percentile_threshold)

###############################################################################

'''This function plots histogram, boxplot and z-score/outlier graphs for the specified column.'''

def outlier_inspect(df, col, min_z = 1, max_z = 5, step = 0.05, max_hist = None, bins = 50):
    plt.figure(figsize=(22, 4))
    plt.suptitle(col, fontsize=16)
    plt.subplot(1,3,1)
    if max_hist == None:
        sns.distplot(df[col], kde=False, bins = 50)
    else :
        sns.distplot(df[df[col]<=max_hist][col], kde=False, bins = 50)
    plt.subplot(1,3,2)
    sns.boxplot(df[col])
    plt.subplot(1,3,3)
    z_score_inspect = outlier_zscore(df, col, min_z = min_z, max_z = max_z, step = step)
    plt.show()

###############################################################################

"""This function gives max/min threshold, number of data, number of outlier and plots its boxplot,
according to the tree type and the entered z-score value for the relevant column."""

def num_outliers(df, target, col, whis=1.5):
    q1 = df.groupby(target)[col].quantile(0.25)
    q3 = df.groupby(target)[col].quantile(0.75)
    iqr = q3 - q1
    print("Column Name:", col)
    print("whis:", whis)
    print("-------------------------------------------")
    for i in np.sort(df[target].unique()):
        min_threshold = q1.loc[i] - whis*iqr.loc[i]
        max_threshold = q3.loc[i] + whis*iqr.loc[i]
        print("min_threshold:", min_threshold, "\nmax_threshold:", max_threshold)
        num_outliers = len(df[df[target]==i][col][(df[col]<min_threshold) | (df[col]>max_threshold)])
        print(f"Num_of_values for {i} :", len(df[df[target]==i]))
        print(f"Num_of_outliers for {i} :", num_outliers)
        print("-------------------------------------------")
    return 

###############################################################################

"""This function assigns the NaN-value first and then drop related rows, according to the tree type and the entered
whis value and plots the boxplot for the relevant column. """

def remove_outliers(df, target, col, whis=1.5):
    q1 = df.groupby(target)[col].quantile(0.25)
    q3 = df.groupby(target)[col].quantile(0.75)
    iqr = q3 - q1
    for i in np.sort(df[target].unique()):
        min_threshold = q1.loc[i] - whis*iqr.loc[i]
        max_threshold = q3.loc[i] + whis*iqr.loc[i]
        df.loc[((df[target]==i) & ((df[col]<min_threshold) | (df[col]>max_threshold))), col] = np.nan
    return 

###############################################################################

def plot_multiclass_roc(model, X_test_scaled, y_test, n_classes, figsize=(5,5)):
    from sklearn.metrics import auc, roc_curve
    import matplotlib.pyplot as plt
    y_score = model.decision_function(X_test_scaled)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
      
def plot_multiclass_roc_for_tree(model, X_test_scaled, y_test, n_classes, figsize=(5,5)):
    from sklearn.metrics import auc, roc_curve
    y_score = model.predict_proba(X_test_scaled)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
###############################################################################################################

def select_dataset(real, simulation):
    from termcolor import colored
    
    training_data_sets = ["real", "sim"]
    training_data_set = input(f"Select the training data set >>> '{training_data_sets[0]}' or '{training_data_sets [1]}'\n")
    print(colored(f'***************************************************',"red", attrs=["bold"]))
    test_data_sets = ["real", "sim"]
    test_data_set = input(f"Select the test data set >>> '{test_data_sets[0]}' or '{test_data_sets [1]}'\n")
    print(colored(f'***************************************************',"red", attrs=["bold"]))
    feature_sets = ["all", "selected"]
    feature_set = input(f"Select the feature_set >>> '{feature_sets[0]}' or '{feature_sets[1]}'\n")
    print(colored(f'***************************************************',"red", attrs=["bold"]))   
    
    if training_data_set == "real" and test_data_set == "real":
        print(colored(f'Selected training data set >>> {training_data_set}\nSelected test data set >>> {test_data_set}', "yellow", attrs=["bold"]))
        print(colored(f'!!! Model will be trained and tested with "{training_data_set}" data set.', "blue", attrs=["bold"]))
        print(colored(f'***************************************************',"red", attrs=["bold"]))
        X_train = real.copy()
        X_test = real.copy()
        whis_train = 3
        n_splits = 10

        # # Drop Columns
        df = X_train
        df.drop(["polar", "temperature"], axis=1, inplace=True)

    elif training_data_set == "real" and test_data_set == "sim":
        print(colored(f'Selected training data set >>> {training_data_set}\nSelected test data set >>> {test_data_set}', "yellow", attrs=["bold"]))
        print(colored(f'!!! Model will be trained with "{training_data_set}" data set and tested with "{test_data_set}" data set.', "blue", attrs=["bold"]))
        print(colored(f'***************************************************',"red", attrs=["bold"]))
        X_train = real.copy()
        whis_train = 3
        n_splits = 10

        X_test = simulation.copy()
        whis_test= 1.5

        # # Drop Columns
        df = X_train
        df.drop(["polar", "temperature"], axis=1, inplace=True)


        df = X_test
        df.drop(["damage_level"], axis=1, inplace=True)

    elif training_data_set == "sim" and test_data_set == "real":
        print(colored(f'Selected training data set >>> {training_data_set}\nSelected test data set >>> {test_data_set}', "yellow", attrs=["bold"]))
        print(colored(f'!!! Model will be trained with "{training_data_set}" data set and tested with "{test_data_set}" data set.', "blue", attrs=["bold"]))
        print(colored(f'***************************************************',"red", attrs=["bold"]))
        X_train = simulation.copy()
        whis_train= 1.5
        n_splits = 10

        X_test = real.copy()
        whis_test = 1.5

        # Drop Columns
        df = X_train
        df.drop(["damage_level"], axis=1, inplace=True)


        df = X_test
        df.drop(["polar", "temperature"], axis=1, inplace=True)


    elif training_data_set == "sim" and test_data_set == "sim":
        print(colored(f'Selected training data set >>> {training_data_set}\nSelected test data set >>> {test_data_set}', "yellow", attrs=["bold"]))
        print(colored(f'!!! Model will be trained and tested with "{training_data_set}" data set.', "blue", attrs=["bold"]))
        print(colored(f'***************************************************',"red", attrs=["bold"]))
        X_train = simulation.copy()
        X_test = simulation.copy()
        whis_train= 1.5
        n_splits = 10

        # Drop Columns
        df = X_train
        df.drop(["damage_level"], axis=1, inplace=True)


    else:
        print(colored(f'Valid training data sets!! >>> {training_data_sets}\n\
Valid test data sets!! >>> {test_data_sets}', "yellow", attrs=["bold"]))
        print(colored(f'Please call the function again to select a valid data set!!!', "blue", attrs=["bold"]))
   
    print(colored(f'***************************************************',"red", attrs=["bold"]))
    print(colored(f'Selected feature set >>> {feature_set}', "yellow", attrs=["bold"]))
    print(colored(f'Valid feature sets!! >>> {feature_sets}', "yellow", attrs=["bold"]))
    print(colored(f'If you selected an invalid feature set, model will be trained with selected features.\n\
Or please call the function again to select a valid feature set!!!', "blue", attrs=["bold"]))
    print(colored(f'***************************************************',"red", attrs=["bold"]))
    try:
        return X_train, X_test, training_data_set, test_data_set, feature_set, whis_train, whis_test, n_splits
    except:
        pass

###############################################################################################################

def model_selection(X_train, y_train):
    # # classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, plot_tree 
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier, plot_importance
    # Logistic Regression
    log = LogisticRegression(random_state=42)  # class_weight="balanced"
    log.fit(X_train, y_train)
    # Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=42)  # class_weight="balanced"
    decision_tree.fit(X_train, y_train)
    # Random Forest
    random_forest = RandomForestClassifier(random_state=42)  # class_weight="balanced"
    random_forest.fit(X_train, y_train)
    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    # SVC
    svc = SVC(random_state=42)  # class_weight="balanced"
    svc.fit(X_train, y_train)
    # XGB
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)
    # AdaBoosting
    ab = AdaBoostClassifier(random_state=42)
    ab.fit(X_train, y_train)
    # GB GradientBoosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)  
    # LGBM 
    lgbm = LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)

    # Model Accuracy on Training Data
    print(f"\033[1m1) Logistic Regression Training Accuracy:\033[0m {log.score(X_train, y_train)}")
    print(f"\033[1m2) Decision Tree Training Accuracy:\033[0m {decision_tree.score(X_train, y_train)}")
    print(f"\033[1m3) Random Forest Training Accuracy:\033[0m {random_forest.score(X_train, y_train)}")
    print(f"\033[1m4) KNN Training Accuracy:\033[0m {knn.score(X_train, y_train)}")
    print(f"\033[1m5) SVC Training Accuracy:\033[0m {svc.score(X_train, y_train)}")
    print(f"\033[1m6) XGBoosting Training Accuracy:\033[0m {xgb.score(X_train, y_train)}")
    print(f"\033[1m7) AdaBoosting Training Accuracy:\033[0m {ab.score(X_train, y_train)}")
    print(f"\033[1m8) GradiendBoosting Training Accuracy:\033[0m {gb.score(X_train, y_train)}")
    print(f"\033[1m8) LGBM Training Accuracy:\033[0m {lgbm.score(X_train, y_train)}")
    return log, decision_tree, random_forest, knn, svc, xgb, ab, gb, lgbm






