



import pandas as pd
import numpy as np
import math
import os
import pre_processing
import upsampling
import bidirectional_feature_selection
from ensamble_model_predictor import ensamble_model_predictions_for_binary_classification as predict_with_ensamble_of_models
from sklearn.ensemble import RandomForestClassifier
import itertools

data = pd.read_csv(os.path.join('data', 'train.csv'))
data.dtypes
data = pre_processing.execute_preprocessing(data, get_dummies=True)


target_column = 'Response'
data[target_column].value_counts()

# best_features = bidirectional_feature_selection.BiDirectionalSelection(data, target_column, 1, clf=RandomForestClassifier(), n_features=70, sequence=['f', 'f'])


# Load best_70_features
import pickle
with open(os.path.join('data', 'pickle', "best_70_features.pickle"), "rb") as _fp:
    best_features = pickle.load(_fp)

# best_features = ['Vehicle_Damage', 'Age', 'Vintage_239', 'Vintage_111', 'Region_Code_43.0', 'Vintage_94', 'Region_Code_41.0', 'Policy_Sales_Channel_120.0', 'Previously_Insured', 'Policy_Sales_Channel_58.0', 'Policy_Sales_Channel_124.0', 'Vintage_171', 'Policy_Sales_Channel_143.0', 'Region_Code_5.0']
best_features = list(set(best_features+['Gender', 'Age', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium'])) + [target_column]

data = data[best_features]



##############################
##############################
####                      ####
####    BEST ALGORITHM    ####
####       PIPELINE       ####
####                      ####
##############################
##############################


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def start_experiment_for_best_classifier_algorithm():
    
    X = data.drop([target_column], axis = 1)
    y = data[target_column]
    
    # execute Up-Sampling
    X, y = upsampling.up_sample_minority_class(X, y, target_column)
    
    
    random_state = 124567
    
    
    X, y = shuffle(X, y, random_state=random_state)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True)
    
    
    # Initialize all models
    
    clf_A = KNeighborsClassifier(n_neighbors=3, n_jobs=-2)
    clf_B = GaussianNB()
    clf_C = DecisionTreeClassifier(max_depth=5)
    clf_D = RandomForestClassifier(random_state=random_state, n_jobs=-2)
    clf_E = SVC(kernel='rbf',probability=True)
    clf_F = XGBClassifier(learnin_rate=0.2, max_depth= 8, n_jobs=-2)
    clf_G = AdaBoostClassifier()
    clf_H = SGDClassifier(loss='log', n_jobs=-2)
    clf_I = LinearSVC()
    clf_J = LogisticRegression(n_jobs=-2)
    clf_K = SVC(kernel='poly', probability=True)
    clf_L = SVC(kernel='sigmoid', probability=True)
    
    classifier_list = [ clf_F, clf_B, clf_H, clf_J, clf_C, clf_D, clf_G ] # clf_A, clf_E, clf_K, clf_L
    classifier_dict = {}
    
    for c in classifier_list:
        classifier_dict[c.__class__.__name__] = c
    
    
    # Calculate the number of samples for 1%, 10%, and 100% of the training data
    samples_100 = len(y_train)
    samples_10 = int(len(y_train)/10)
    samples_1 = int(len(y_train)/100)
    samples_0_1 = int(len(y_train)/1000)
    
    # Collect results on the learners
    results = {}
    classifiers = []
    for clf in classifier_list:
        clf_name = clf.__class__.__name__
        print("clf_name", clf_name)
        results[clf_name] = {}
        sample_set = [ samples_1 ]
        for i, samples in enumerate(sample_set): # 
            # print('y_train_sum_cr_samples', sum(y_train[:samples]))
            # print('len_y_train', len(y_train[:samples]))
            # print('y_train_sum_all', sum(y_train))
            # print("shape: ", type(X_train), type(y_train), type(X_test), type(y_test))
            # print("shape: ", (X_train.shape), (y_train.shape), (X_test.shape), (y_test.shape))
            
            cr_result, clf = train_predict(clf, samples, X_train, y_train, X_test, y_test)
            results[clf_name][i] = cr_result
            if samples == max(sample_set):
                classifiers.append(clf)
        
     
    # ensamble_predictions = predict_with_ensamble_of_models(classifiers, X_test)
    
    
    
    # results['ensamble'] = {}
    
    # results['ensamble']['accuracy_test'] = accuracy_score(y_test, ensamble_predictions)
    # results['ensamble']['precision_test'] = precision_score(y_test, ensamble_predictions)
    # results['ensamble']['recall_test'] = recall_score(y_test, ensamble_predictions)
    # results['ensamble']['fbeta_test'] = fbeta_score(y_test, ensamble_predictions, beta=0.5)
    
    
    return results, classifiers, X_test, y_test, classifier_dict




from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import cross_val_score
from time import time
def train_predict(clf, sample_size, X_train, y_train, X_test, y_test):
    
    # X = data.drop([target_column], axis = 1)
    # y = data[target_column]
    
    
    thresholds = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    all_results = {}

    results = {}
    
    # Fit the classifier to the training data 
    # Gets start time
    start = time()
    clf = clf.fit(X_train[:sample_size], y_train[:sample_size])
    # Gets end time
    end = time()
    
    # Calculates the training time
    results['train_time'] = end - start 
    
    # Get the predictions on the test set
    # Gets start time
    start = time()
    # predictions_test = clf.predict(X_test)
    # predictions_train = clf.predict(X_train)
    
    # Set predictions based on the selected threshold on probablity of preddictions
    for threshold in thresholds:
        
        
        
        prediction_probablities = clf.predict_proba(X_test)
        predictions_test = (prediction_probablities [:,1] >= threshold).astype('float')
        prediction_probablities = clf.predict_proba(X_train)
        predictions_train = (prediction_probablities [:,1] >= threshold).astype('float')
    
        
        # Gets end time
        end = time()
        
        # Calculate the total prediction time
        results['threshold'] = threshold
        results['pred_time'] = end - start
        
        # Compute accuracy on the first 300 training samples 
        results['accuracy_train'] = accuracy_score(y_train, predictions_train)
        results['precision_train'] = precision_score(y_train, predictions_train)
        results['recall_train'] = recall_score(y_train, predictions_train)
        results['fbeta_train'] = fbeta_score(y_train, predictions_train, beta=0.5)
        
        # Compute accuracy on test set using accuracy_score()
        results['accuracy_test'] = accuracy_score(y_test, predictions_test)
        results['precision_test'] = precision_score(y_test, predictions_test)
        results['recall_test'] = recall_score(y_test, predictions_test)
        results['fbeta_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
        
        
        all_results[threshold] = results

    # print the trained message
    print("{} trained on {} samples.".format(clf.__class__.__name__, sample_size))
    
    highest_fbeta_test = 0.0
    highest_performing_threshold = -1
    for cr_res_key in all_results.keys():
        if all_results[cr_res_key]['fbeta_test'] >= highest_fbeta_test:
            highest_performing_threshold = cr_res_key
            highest_fbeta_test = all_results[cr_res_key]['fbeta_test']
    
    
    # Return the results
    return all_results[highest_performing_threshold], clf



best_classifier_results, classifiers, X_test, y_test, classifier_dict = start_experiment_for_best_classifier_algorithm()

best_classifier_results



ensamble_predictions = predict_with_ensamble_of_models([classifiers[0],classifiers[3],classifiers[5],classifiers[6], classifiers[8]], X_test, weights=[8,4,5,3,3], thresholds=[0.75,0.75,0.75,0.75,0.75])



ensamble_results = {}

ensamble_results['ensamble_accuracy_test'] = accuracy_score(y_test, ensamble_predictions)
ensamble_results['ensamble_precision_test'] = precision_score(y_test, ensamble_predictions)
ensamble_results['ensamble_recall_test'] = recall_score(y_test, ensamble_predictions)
ensamble_results['ensamble_fbeta_test'] = fbeta_score(y_test, ensamble_predictions, beta=0.5)

ensamble_results['ensamble_fbeta_test']



##############################
##############################
####                      ####
####   PARAMETER TUNING   ####
####                      ####
##############################
##############################





best_classifiers = pd.Series({c: info[0]['fbeta_test'] for c,info in best_classifier_results.items()}).sort_values(ascending=False)
best_classifiers = best_classifiers[best_classifiers>0.64][:5]





from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

def ExecuteGridSearchCV(clf = AdaBoostClassifier(), parameters=None):
    
    
    X = data.drop([target_column], axis = 1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, shuffle=True)

    fbeta_scorer = make_scorer(fbeta_score, beta=0.5)
    
    if parameters == None:
        parameters = { 'random_state': [55]} # 'n_estimators':[30,50,80,120], 'learning_rate':[0.2,0.5,1.0,1.25,2.0], 'algorithm':['SAMME', 'SAMME.R'],
    
    
    grid_clf = GridSearchCV(clf, param_grid=parameters, scoring=fbeta_scorer)
    grid_clf.fit(X_train, y_train)
    
    return grid_clf, grid_clf.score(X_test, y_test)


grid_clf, best_result = ExecuteGridSearchCV(AdaBoostClassifier())
best_result




def get_classifier_with_parameters(clf_name, param):
    if clf_name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(max_depth=param[0], min_samples_split=param[1], criterion=param[2])
    elif clf_name == 'RandomForestClassifier':
        return RandomForestClassifier(max_depth=param[0], min_samples_split=param[1], criterion=param[2], n_estimators=param[3], n_jobs=-2)
    elif clf_name == 'XGBClassifier':
        return XGBClassifier(max_depth=param[0], n_estimators=param[1], n_jobs=-2)
    elif clf_name == 'GaussianNB':
        return GaussianNB(var_smoothing=param[0])
    elif clf_name == 'SGDClassifier':
        return SGDClassifier(loss=param[0], penalty=param[1], alpha=param[2], n_jobs=-2)


parameters = {}

parameters['DecisionTreeClassifier'] =  [ [9,18,30,40,60], [3,5,8], ['gini', 'entropy'] ]
parameters['RandomForestClassifier'] =  [ [9,18,30,40,60], [3,5,8], ['gini', 'entropy'], [60, 90, 120, 180] ]
parameters['XGBClassifier'] =  [ [9,18,24,30,40], [60, 90, 120, 180] ]
parameters['GaussianNB'] =  [ np.logspace(0,-9, num=10) ]
parameters['SGDClassifier'] =  [ [ 'log', 'modified_huber'], ['l2', 'l1', 'elasticnet'], np.logspace(0,-6, num=7)  ]

thresholds = [0.40, 0.50, 0.60, 0.75, 0.85]





def CustomGridSearch(X, y, clf_name, parameters, thresholds):
    
    params = [e for e in itertools.product(*parameters[clf_name], thresholds)]
    
    print("total params: ", len(params), "\n")
    
    results = np.zeros(len(params))
    classifiers = []
    
    for i, param in enumerate(params):
        if i % 20 == 0:
            print(i)
        
        grid_clf = get_classifier_with_parameters(clf_name, param)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, shuffle=True)
    
        grid_clf.fit(X_train, y_train)
        
        prediction_probablities = grid_clf.predict_proba(X_test)
        predictions_test = (prediction_probablities [:,1] >= param[-1]).astype('float')
        
        results[i] = fbeta_score(y_test, predictions_test, beta=0.5)
        classifiers.append(grid_clf)
    
    best_clf = classifiers[np.argmax(results)]
        
    return (max(results), params[np.argmax(results)], best_clf)

        
def get_best_of_classifiers_via_custom_grid_search(best_classifiers, target_column, data, parameters, thresholds):
    data_min = data.iloc[:int(data.shape[0]/100),:]
    
    X = data_min.drop([target_column], axis = 1)
    y = data_min[target_column]
    
    # execute Up-Sampling
    X, y = upsampling.up_sample_minority_class(X, y, target_column)
    random_state = 124567
    X, y = shuffle(X, y, random_state=random_state)
    
    grid_search_results_for_best_classifiers = {}
    
    for clf_name in best_classifiers.keys():
        if clf_name != 'RandomForestClassifier' or True:
            print("\n", clf_name, "\n")
            grid_search_results_for_best_classifiers[clf_name] = CustomGridSearch(X, y, clf_name, parameters, thresholds)
        
    return grid_search_results_for_best_classifiers


grid_search_results_for_best_classifiers = get_best_of_classifiers_via_custom_grid_search(best_classifiers, target_column, data, parameters, thresholds)





ensamble_predictions = predict_with_ensamble_of_models([c[2] for i,c in grid_search_results_for_best_classifiers.items()], X_test, weights=[10,10,10,10,10], thresholds=[c[1][-1] for i,c in grid_search_results_for_best_classifiers.items()])

ensamble_results = {}

ensamble_results['ensamble_accuracy_test'] = accuracy_score(y_test, ensamble_predictions)
ensamble_results['ensamble_precision_test'] = precision_score(y_test, ensamble_predictions)
ensamble_results['ensamble_recall_test'] = recall_score(y_test, ensamble_predictions)
ensamble_results['ensamble_fbeta_test'] = fbeta_score(y_test, ensamble_predictions, beta=0.5)

ensamble_results['ensamble_fbeta_test']




from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

def get_results_with_random_forest(data):
    X = data.drop([target_column], axis = 1)
    y = data[target_column]
    
    
    
    
    accuracy_score = []
    precision_score = []
    recall_score =  []
    fbeta_score = []
    
    for i in range(100):
    
        stratified_folds = StratifiedKFold(n_splits=10, shuffle=True)
        
        for train_indices, test_indices in stratified_folds.split(X, y): 
                X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
                y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
                clf = RandomForestClassifier(n_jobs=-2, class_weight='balanced')
                clf.fit(X_train, y_train)
                predictions = y.copy()
                predictions[test_indices] = clf.predict(X_test)
        
        
        
        cr_accuracy_score = accuracy_score(y, predictions)
        cr_precision_score = precision_score(y, predictions)
        cr_recall_score =  recall_score(y, predictions)
        cr_fbeta_score = fbeta_score(y, predictions, beta=0.5)
        
        
        accuracy_score.append(cr_accuracy_score)
        precision_score.append(cr_precision_score)
        recall_score.append(cr_recall_score)
        fbeta_score.append(cr_fbeta_score)
        

    print("cr_accuracy_score = ", np.mean(accuracy_score))
    print("cr_fbeta_score = ", np.mean(precision_score))
    print("cr_recall_score = ", np.mean(recall_score))
    print("cr_precision_score = ", np.mean(fbeta_score))




def get_results_with_xgb(data):
    X = data.drop([target_column], axis = 1)
    y = data[target_column]
    
    
    
    accuracy_score = []
    precision_score = []
    recall_score =  []
    fbeta_score = []
    
    for i in range(100):
    
        X, y = upsampling.up_sample_minority_class(X, y, target_column)
        random_state = 124567
        X, y = shuffle(X, y, random_state=random_state)
        
        stratified_folds = StratifiedKFold(n_splits=10, shuffle=True)
        
        for train_indices, test_indices in stratified_folds.split(X, y): 
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            clf = XGBClassifier(max_depth=9, n_estimators=180, n_jobs=-2)
            clf.fit(X_train, y_train)
            predictions = y.copy()
            predictions[test_indices] = clf.predict(X_test)
        
        
        
        cr_accuracy_score = accuracy_score(y, predictions)
        cr_precision_score = precision_score(y, predictions)
        cr_recall_score =  recall_score(y, predictions)
        cr_fbeta_score = fbeta_score(y, predictions, beta=0.5)
        
        
        accuracy_score.append(cr_accuracy_score)
        precision_score.append(cr_precision_score)
        recall_score.append(cr_recall_score)
        fbeta_score.append(cr_fbeta_score)
        

    print("cr_accuracy_score = ", np.mean(accuracy_score))
    print("cr_fbeta_score = ", np.mean(precision_score))
    print("cr_recall_score = ", np.mean(recall_score))
    print("cr_precision_score = ", np.mean(fbeta_score))
    


get_results_with_random_forest(data)

get_results_with_xgb(data)


##############################################
##############################################
####                                      ####
####            --------------            ####
####            --- RESULT ---            ####
####            --------------            ####
####                                      ####
####      BEST PERFORMING CLASSIFIER      ####
####          XGBOOST CLASSIFIER          ####
####                                      ####
####        10-FOLD CROSS VALIDATION      ####
####          WITH 100 ITERATIONS         ####
####         F-BETA SCORE (ß=0.5):        ####
####               0.97544                ####
####                                      ####
##############################################
##############################################

# With 11 features without Dummy Variables
# cr_accuracy_score =  0.9735298825270461
# cr_fbeta_score =  0.9267431170464993
# cr_recall_score =  0.8200813530293299
# cr_precision_score =  0.9578894723680921


# With 502 features and no feature selection
# cr_accuracy_score =  0.965280062202817
# cr_fbeta_score =  0.9546184017659021
# cr_recall_score =  0.9854959777505308
# cr_precision_score =  0.9471989882440861


# With 72 selected features
# cr_accuracy_score =  0.9834045556356329
# cr_fbeta_score =  0.975442445919053
# cr_recall_score =  0.997604657908666
# cr_precision_score =  0.970054900318701


##############################################
##############################################
####                                      ####
####         ---------------              ####
####         ---  BONUS  ---              ####
####         ---------------              ####
####                                      ####
####     FEATURE IMPORTANCE:              ####
####                                      ####
####     1) Region_Code_29.0              ####
####     2) Vintage_189                   ####
####     3) Policy_Sales_Channel_25.0     ####
####     4) Policy_Sales_Channel_13.0     ####
####     5) Policy_Sales_Channel_30.0     ####
####     6) Vintage_209                   ####
####     7) Vintage_37                    ####
####     8) Vintage_62                    ####
####     9) Vintage_136                   ####
####     10) Vintage_276                  ####
####     11) Policy_Sales_Channel_1.0     ####
####     12) Previously_Insured           ####
####                                      ####
####                                      ####
##############################################
##############################################


# Feature Selection with Boruta

from boruta import BorutaPy 


X = data.drop([target_column], axis = 1)
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True)
    

clf = RandomForestClassifier(n_jobs=-2)



# define Boruta feature selection method
feat_selector = BorutaPy(clf, n_estimators='auto', verbose=2, alpha=0.20)
 
# find all relevant features
feat_selector.fit(X.values, y.values)
 
# check selected features
feat_selector.support_
 
# check ranking of features
feat_selector.ranking_


pd.Series(data.iloc[:,:-1].columns.tolist())[feat_selector.support_]



##############################################
##############################################
####                                      ####
####       ENSAMBLE OF BEST MODELS        ####
####                                      ####
##############################################
##############################################














##############################
##############################
####                      ####
####      REFERENCES      ####
####                      ####
##############################
##############################



# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.pie.html
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html
# https://python-graph-gallery.com/161-custom-matplotlib-donut-plot/
# https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
# https://stackoverflow.com/questions/59559682/how-to-change-pandas-dataframe-plot-fontsize-of-xlabel
# https://stackoverflow.com/questions/57958432/how-to-add-table-title-in-python-preferably-with-pandas
# https://stackoverflow.com/questions/31948879/using-explicit-predefined-validation-set-for-grid-search-with-sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html