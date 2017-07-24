import pandas as pd
import numpy as np

def features_score(select, X):

    """ Tabulate the scores of the selection against features names
    
    Args:
        
        select: SelectKBest.fit object
        
        X: Dataframe of the features and values
        
    Returns:
        
        output: Panda dataframe with the feature name and its SelectKBest score 
        
    """

    a = np.array(X.columns)*select.get_support()
    b = select.scores_*select.get_support()
    short_features = [a[i] for i in a.nonzero()[0]]
    short_scores = [b[i] for i in a.nonzero()[0]]
    output = pd.DataFrame(data = short_scores,
                            index = short_features,
                            columns = ['scores'])
    
    return output
    
def transform(select,X):

    """ Transform the original input matrix into an output matrix with 
        just the selected features
    
    Args:
        
        select: SelectKBest.fit object
        
        X: Dataframe of the input matrix of X
        
    Returns:
        
        output: Panda dataframe of X with just the selected features
        
    """

    output = pd.DataFrame(select.transform(X))
    a = np.array(X.columns)*select.get_support()
    short_features = [a[i] for i in a.nonzero()[0]]
    output.index = X.index
    output.columns = short_features
    return output

def svc_tune(X, y, tune):
    
    """ Creates estimator object
    
    Args:
        
        X: Dataframe of the features and values

        y: Dataframe of the labels 

        tune: Boolean variable where True means estimator is tuned whilst
        False mean that estimator is not tuned
        
    Returns:
        
        output: Estimator object
        
    """

    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    
   
    svc = LinearSVC(random_state = 42)
    
    if tune ==True:
        pipe = Pipeline([
            ('svc',svc)
        ])
        
        parameters = dict(svc__C=[0.2, 0.4, 0.6, 0.8, 1,1.2,1.4,1.6,1.8],
                            svc__tol = [1e-2, 1e-3, 1e-4])
        
        cv = GridSearchCV(pipe, param_grid=parameters)
        
        return cv.fit(X,y)
    else:
        return svc.fit(X,y)

def gaussian_tune(X, y, tune):
    
    """ Creates estimator object
    
    Args:
        
        X: Dataframe of the features and values

        y: Dataframe of the labels 

        tune: Boolean variable where True means estimator is tuned whilst
        False mean that estimator is not tuned
        
    Returns:
        
        output: Estimator object
        
    """

    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import GaussianNB

    gaussian = GaussianNB()
    if tune == True: 
        pipe = Pipeline([
            ('gaussian',gaussian)
        ])
        
        parameters = dict()
        
        cv = GridSearchCV(pipe, param_grid=parameters)
    
        return cv.fit(X,y)
    else:
        return gaussian.fit(X,y)

def log_tune(X, y, tune):
    
    """ Creates estimator object
    
    Args:
        
        X: Dataframe of the features and values

        y: Dataframe of the labels 

        tune: Boolean variable where True means estimator is tuned whilst
        False mean that estimator is not tuned
        
    Returns:
        
        output: Estimator object
        
    """

    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegressionCV
    log = LogisticRegressionCV(random_state = 42)
    
    if tune == True:
        pipe = Pipeline([
            ('log',log)
        ])
        
        parameters = dict(log__Cs = [2, 4, 8, 10])
        
        cv = GridSearchCV(pipe, param_grid=parameters)
        
        return cv.fit(X,y)
    else:
        return log.fit(X,y)

def rfc_tune(X, y, tune):
    
    """ Creates estimator object
    
    Args:
        
        X: Dataframe of the features and values

        y: Dataframe of the labels 

        tune: Boolean variable where True means estimator is tuned whilst
        False mean that estimator is not tuned
        
    Returns:
        
        output: Estimator object
        
    """

    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    
    rfc =  RandomForestClassifier(random_state = 42)
    if tune == True:
        pipe = Pipeline([
            ('rfc',rfc)
        ])
        
        parameters = dict(rfc__n_estimators = [3,5,10,15],
                            rfc__max_features = ["auto","sqrt","log2",None],
                            rfc__criterion = ["gini","entropy"])
        
        cv = GridSearchCV(pipe, param_grid=parameters)
        
        return cv.fit(X,y)
    else:
        
        return rfc.fit(X,y)

def ada_tune(X, y, tune):
    
    """ Creates estimator object
    
    Args:
        
        X: Dataframe of the features and values

        y: Dataframe of the labels 

        tune: Boolean variable where True means estimator is tuned whilst
        False mean that estimator is not tuned
        
    Returns:
        
        output: Estimator object
        
    """

    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import AdaBoostClassifier
    ada =  AdaBoostClassifier(random_state = 42)
    if tune == True:
        pipe = Pipeline([
            ('ada',ada)
        ])
        
        parameters = dict(ada__learning_rate = [0.2,0.4,0.6,0.8,1])
        
        cv = GridSearchCV(pipe, param_grid=parameters)
        
        return cv.fit(X,y)
    else:

        return ada.fit(X,y)
        
def print_score(cv, features_test, labels_test):

    """ Print the recall and precision score of the estimator
    
    Args:
        
        cv: Estimator object

        features_test: Dataframe of the test features and values 

        labels_test: Dataframe of the test labels 
        
    Returns:
        
        output: Estimator object
        
    """

    from sklearn.metrics import ( 
        precision_score,
        recall_score)
    print "Recall score: " + str(recall_score(labels_test, cv.predict(features_test)))
    print "Precision score: " + str(precision_score(labels_test, cv.predict(features_test)))
    #print "Best estimator parameters: " + str(cv.best_estimator_)
    
