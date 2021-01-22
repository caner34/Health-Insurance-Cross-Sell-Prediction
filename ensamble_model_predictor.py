
import numpy as np

def ensamble_model_predictions_for_binary_classification(classifiers, X, weights=[], thresholds=[]):
    
    predictions = []
    
    
    if len(thresholds) == 0:
        weights = len(classifiers) * [0.5]
    
    for clf, threshold in zip(classifiers,thresholds):
        prediction_probablities = clf.predict_proba(X)
        prediction = (prediction_probablities [:,1] >= threshold).astype('float')
        predictions.append(prediction)
    
    if len(weights) == 0:
        weights = len(classifiers) * [1]
        
    
    if sum(weights) != 1:
        sum_weights = sum(weights)
        for i,w in enumerate(weights):
            weights[i] = w / sum_weights
    
    for i, (p,w) in enumerate(zip(predictions, weights)):
        predictions[i] = p * w
    
    return np.round(sum(predictions))
