import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score



def sequential_forward_selection(data, beatified, excommunicated, target, n_iters, clf):
    on_as_sirat = [c for c in data.columns.tolist() if c not in beatified and c not in excommunicated and c not in [target]]
    
    if len(on_as_sirat) == 0:
        return beatified
    
    accuricies = np.zeros(len(on_as_sirat))
    
    for c_index, c in enumerate(on_as_sirat):
        on_katabasis = beatified + [c]
        
        X = data[on_katabasis]
        y = data[target]
        
        n_iters = n_iters
        cr_performances =  np.zeros(n_iters)
        
        for i in range(n_iters):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, stratify=y, shuffle=True)

            #naive = GaussianNB()
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            cr_performances[i] = accuracy_score(y_test, predictions)
            
    
        accuricies[c_index] = cr_performances.mean()

    beatified.append(on_as_sirat[accuricies.argmax()])
    
    return beatified



def sequential_backward_selection(data, beatified, excommunicated, target, n_iters, clf):
    by_avernus = [c for c in data.columns.tolist() if c not in beatified and c not in excommunicated and c not in [target]]
    
    if len(by_avernus) == 0:
        return excommunicated
    
    accuricies = np.zeros(len(by_avernus))
    
    for c_index, c in enumerate(by_avernus):
        on_katabasis = beatified + by_avernus
        on_katabasis = [f for f in on_katabasis if f != c]
        
        X = data[on_katabasis]
        y = data[target]
        
        n_iters = n_iters
        cr_performances =  np.zeros(n_iters)
        
        for i in range(n_iters):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, stratify=y, shuffle=True)

            # naive = GaussianNB()
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            cr_performances[i] = accuracy_score(y_test, predictions)
            
    
        accuricies[c_index] = cr_performances.mean()

    excommunicated = [by_avernus[accuricies.argmin()]] + excommunicated
    
    return excommunicated

def BiDirectionalSelection(data, target, n_iters, clf, n_features=0, sequence=['f', 'b']):
    
    print("BiDirectionalSelection started n_columns: ", len(data.columns.tolist()))
    
    sequence = sequence * int((len(data.columns.tolist()) + 1) / len(sequence))
    
    beatified = []
    excommunicated = []
    
    for step in sequence:
        
        
        endymion = [c for c in data.columns.tolist() if c not in beatified + excommunicated + [target]]
        if len(endymion) == 0:
            break
        
        if n_features != 0 and len(beatified) == n_features:
            return beatified
        
        if step == 'f':
            beatified = sequential_forward_selection(data, beatified, excommunicated, target, n_iters, clf)
            #print("beatified: ",beatified)
        elif step == 'b':
            excommunicated = sequential_backward_selection(data, beatified, excommunicated, target, n_iters, clf)
            #print("excommunicated: ",excommunicated)
            
        print('len(beatified): ', len(beatified))
    
    final_sorting = beatified + excommunicated
    if n_features != 0:
        return final_sorting[:n_features]
    else:
        return final_sorting
    
