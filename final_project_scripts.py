import numpy as np
import pandas as pd

def featureFormatpd(dataframe, features, zero_NaN=True):
    """Choose features for analysis and replace np.nan into 0
    
    Args:
        
        dataframe: Panda dataframe with all the data of to be examined
        
        features: List with all the features to be examined
        
        zero_NaN: Boolean to indicate whether to replace np.nan with zeros
    
    Returns:
        
        output: Panda dataframe with the selected features
        
    """
    
    output = pd.DataFrame()
    for feature in features:
        try:
            output[feature] = dataframe[feature]
        except KeyError:
            print "error: key ", feature, " not present"
            return
    if zero_NaN:
        output = output.fillna(0)
    return output


def targetFeatureSplitpd(dataframe, labelname ):
    """Split dataframe into features and labels
    
    Args:
        
        dataframe: Panda dataframe with all the data of to be examined
        
        labelname: String of the label name
    
    Returns:
        
        target: dataframe with only a single Y variable
        
        features: dataframe with all the X variables
        
    """

    target = dataframe[labelname]
    features = dataframe.drop(labelname,axis = 1)

    return target, features


def nparray_to_dataframe(nparray,index,columns):
    """Convert np.array into dataframe with appropriate index and column names
    
    Args:
        
        nparray: np.array with all of the data to be examined
        
        index: Index object with the list of index names
        
        columns: Index object
    
    Returns:
        
        features: Panda dataframe with all of the data to be examined

    """
    
    import pandas as pd
    features = pd.DataFrame(nparray)
    features.index = index
    features.columns = columns
    return features


def correct_records(dataframe):
    '''Manual correct of 2 records as they are different from the PDF record
    
    Args:

        dataframe: Panda dataframe with all of the data to be examined
    
    Returns:
        
        dataframe: Panda dataframe with two records corrected
 
    '''
    dataframe = dataframe.transpose()
    dataframe['BELFER ROBERT'] = pd.Series({'bonus': np.nan,
                              'deferral_payments': np.nan,
                              'deferred_income': -102500,
                              'director_fees': 102500,
                              'email_address': np.nan,
                              'exercised_stock_options': np.nan,
                              'expenses': 3285,
                              'from_messages': np.nan,
                              'from_poi_to_this_person': np.nan,
                              'from_this_person_to_poi': np.nan,
                              'loan_advances': np.nan,
                              'long_term_incentive': np.nan,
                              'other': np.nan,
                              'poi': False,
                              'restricted_stock': -44093,
                              'restricted_stock_deferred': 44093,
                              'salary': np.nan,
                              'shared_receipt_with_poi': np.nan,
                              'to_messages': np.nan,
                              'total_payments': 3285, 
                              'total_stock_value': np.nan})

    dataframe['BHATNAGAR SANJAY'] = pd.Series({'bonus': np.nan,
                                 'deferral_payments': np.nan,
                                 'deferred_income': np.nan,
                                 'director_fees': np.nan,
                                 'exercised_stock_options': 15456290,
                                 'expenses': 137864,
                                 'from_messages': 29,
                                 'from_poi_to_this_person': 0,
                                 'from_this_person_to_poi': 1,
                                 'loan_advances': np.nan,
                                 'long_term_incentive': np.nan,
                                 'other': np.nan,
                                 'poi': False,
                                 'restricted_stock': 2604490,
                                 'restricted_stock_deferred': -2604490,
                                 'salary': np.nan,
                                 'shared_receipt_with_poi': 463,
                                 'to_messages': 523,
                                 'total_payments': 137864,
                                 'total_stock_value': 15456290} )
    return dataframe.transpose()
    
def select_features_svc(X, y):
    
    """ Creates estimator object
    
    Args:
        
        X: Dataframe of the features and values

        y: Dataframe of the labels 
        
    Returns:
        
        output: Estimator object
        
    """
    from sklearn.feature_selection import SelectKBest, chi2, f_classif
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    svc = LinearSVC(random_state = 123)
    selection = SelectKBest()
    pipe = Pipeline([
        ('select',selection),
        ('svc',svc)
    ])
    
    parameters = dict(select__k = [8,12,16,20])
    
    cv = GridSearchCV(pipe, param_grid=parameters)
    
    return cv.fit(X,y)
