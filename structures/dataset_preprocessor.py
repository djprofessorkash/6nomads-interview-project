# !python3

import numpy as np
import pandas as pd

class Dataset_Preprocessor(object):
    """ Class object instance for preprocessing and cleaning 6Nomads data for predictive analytics. """
    def __init__(self):
        """ Initializer method for object instance creation. """
        self.REL_PATH_TO_EXT_DATA_TRAIN = "../data/external/train.csv"
        self.REL_PATH_TO_EXT_DATA_TEST = "../data/external/test.csv"
        
    def load_data(self, which="both"):
        """ 
        Instance method to load in dataset(s) into conditionally separated/joined Pandas DataFrame(s). 
        
        INPUTS:
            {which}:
                - str(both): Reads in training and testing data files as tuple of individual DataFrames. (DEFAULT)
                - str(all): Reads in training and testing data files as single conjoined DataFrame.
                - str(train): Reads in training data file as single DataFrame.
                - str(test): Reads in testing data file as single DataFrame.
                
        OUTPUTS:
            pandas.DataFrame: Single or multiple Pandas DataFrame object(s) containing relevant data.
        """
        # Validate conditional data loading arguments
        if which not in ["all", "both", "train", "test"]:
            raise ValueError("ERROR: Inappropriate value passed to argument `which`.\n\nExpected value in range:\n - all\n - both\n - train\n - test\n\nActual:\n - {}".format(which))
        
        # Independently load training data
        if which == "train":
            return pd.read_csv(self.REL_PATH_TO_EXT_DATA_TRAIN)
        
        # Independently load testing data
        if which == "test":
            return pd.read_csv(self.REL_PATH_TO_EXT_DATA_TEST)
        else:
            df_train = pd.read_csv(self.REL_PATH_TO_EXT_DATA_TRAIN)
            df_test = pd.read_csv(self.REL_PATH_TO_EXT_DATA_TEST)
            
            # Load merged training and testing data
            if which == "all":
                return pd.concat([df_train, df_test], keys=["train", "test"], sort=True)
            
            # Load separated training and testing data (DEFAULT)
            if which == "both":
                return df_train, df_test
            
    def null_imputer(self, datasets, subset=None, method="fill", na_replace=-1.0, na_filter="all"):
        """
        Instance method to modify (replace or remove) occurrent null (`np.nan`) values across an entire dataset.
        
        INPUTS:
            {datasets}:
                - list: List of datasets; used for iterative data manipulation.
                - pd.DataFrame: Single dataset; converted into list for general data manipulation.
            {subset}:
                - NoneType: If no argument passed, null value imputation occurs for all features in data. (DEFAULT: None)
                - list: Array of feature names to iterate through for null value modification.
            {method}:
                - str(fill): Modifies data entries containing null value occurrences. 
                - str(drop): Removes data entries containing null value occurrences. 
            {na_replace}:
                - NoneType(None): Removes all occurrences of null values across dataset(s).
                - int: Replaces all occurrences of null values with given integer argument.
                - float: Replaces all occurrences of null values with given float argument. (DEFAULT: -1.0)
                - str(mode): Replaces all occurrences of null values with most common values in current sample.
                - str(mean): Replaces all occurrences of null values with average of values in current sample.
                - str(knn): Replaces all occurrences of null values with kNN-approximated data in current sample. (NOT IMPLEMENTED)
            {na_filter}:
                - str(any): Drops data sample if any feature contains null values.
                - str(all): Drops data sample only if all features contain null values.
                
        OUTPUTS:
            NoneType: Null value modification is performed inplace and does not return new object(s).
        """
        # Validation for method instantiation with single passed DataFrame on `datasets` argument.
        if type(datasets) is not list:
            datasets = [datasets]
            
        # Feature-wise inplace null value modification using given replacement data.
        if method == "fill":
            for dataset in datasets:
                if not subset:
                    if na_replace == "mode":
                        dataset.T.fillna(dataset.mode(axis=1)[0], inplace=True)
                    if na_replace == "mean":
                        dataset.T.fillna(dataset.mean(axis=1), inplace=True)
                    else:
                        for feature in dataset:
                            dataset[feature].fillna(na_replace, inplace=True)
                else:
                    if na_replace == "mode":
                        # NOTE: Modifies all null values with mode of given subset of data
                        dataset.T.fillna(dataset[subset].mode(axis=1)[0], inplace=True)
                    if na_replace == "mean":
                        # NOTE: Modifies all null values with mean of given subset of data
                        dataset.T.fillna(dataset[subset].mean(axis=1), inplace=True)
                    else:
                        for feature in subset:
                            if feature in dataset:
                                dataset[feature].fillna(na_replace, inplace=True)
        elif method == "drop":
            for dataset in datasets:
                if not subset:
                    for feature in dataset:
                        dataset[feature].dropna(how=na_filter, inplace=True)
                else:
                    dataset.dropna(how=na_filter, subset=subset, inplace=True)
        print("Null imputation has successfully completed.")
        
    def feature_encoder(self, datasets, target, lookup_table, dtype="discrete", drop_og=False):
        """ 
        Instance method to iteratively encode labels in dataset as numerically categorical data.
        
        INPUTS:
            {datasets}:
                - pd.DataFrame: Single dataset; cast to list for iterative feature mapping.
                - list: List of datasets; used for iterative feature mapping.
            {target}:
                - str: Name of target feature in dataset containing labels on which to encode. 
            {lookup_table}:
                - dict: Encoding table with unencoded data ranges as values and encoded numerical categories as keys.
            {dtype}:
                - str(discrete): Data type parameter; indicates presence of discretized values across dataset. (DEFAULT)
                - str(continuous): Data type parameter; indicates presence of continuous values across dataset.
            {drop_og}:
                - bool(True): Dataset drops original feature after encoding.
                - bool(False): Dataset does not drop original feature after encoding. (DEFAULT)
            
        OUTPUTS:
            NoneType: Dataset insertion is performed inplace and does not return new object(s).
        """
        if type(datasets) is not list:
            datasets = [datasets]
            
        def _encoder_helper(label, lookup_table, dtype):
            """
            Custom helper function to replace unencoded label with encoded value from custom lookup table.

            INPUTS:
                {label}:
                    - int: Unencoded value within Pandas Series to alter to categorically encoded label.
                {lookup_table}:
                    - dict: Encoding table with unencoded data ranges as values and encoded numerical categories as keys.
                {dtype}:
                    - str(discrete): Data type parameter; indicates presence of discretized labels.
                    - str(continuous): Data type parameter; indicates presence of continuous labels.

            OUTPUTS:
                int: Encoded numerical category as new label. (DEFAULT)
                str: Encoded string-based category as new label.
            """
            for key, value in lookup_table.items():
                if dtype == "discrete":
                    if label in value:
                        return key
                if dtype == "continuous":
                    if value[0] <= label < value[1]:
                        return key
        
        encoded_feature = "{}_encoded".format(target)
        for dataset in datasets:
            if encoded_feature in dataset:
                dataset.drop(columns=[encoded_feature], inplace=True)
                
            features = dataset.columns.tolist()
            dataset.insert(loc=features.index(target) + 1, 
                           column="{}_encoded".format(target), 
                           value=dataset[target].apply(_encoder_helper, 
                                                       lookup_table=lookup_table,
                                                       dtype=dtype))
            if drop_og:
                dataset.drop(columns=[target], inplace=True)
        return
    
    def save_dataset(self, dataset, savepath, filetype="csv"):
        """
        Instance method to save current state of dataset to data file accessible by navigating the parent directory.
        
        INPUTS:
            {dataset}:
                - pd.DataFrame: Single parent dataset; used for data formatting and allocation (save to memory).
            {savepath}:
                - str: Relative path location within parent directory to which dataset is saved.
            {filetype}:
                - str(csv): Data formatting type as which dataset is saved. (DEFAULT)
                
        OUTPUTS:
            NoneType: Saving data to memory is performed outside the context of the object and does not return new object(s).
        """
        # Sanitization for method instantiation if unknown value is passed to `filetype` keyword argument
        if filetype not in ["csv", "excel"]:
            raise ValueError("Value passed to keyword argument `filetype` is uninterpretable. EXPECTED: ['csv', 'excel']. ACTUAL: ['{}']".format(filetype))
        
        # Explicit relative pathway declaration and saving process performed on dataset
        savepath += ".{}".format(filetype)
        if filetype == "csv":
            dataset.to_csv(savepath)
        elif filetype == "excel":
            dataset.to_excel(savepath)
        return