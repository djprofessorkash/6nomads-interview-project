# !python3

import numpy as np
import pandas as pd

class Dataset_Processor(object):
    """ Class object instance for processing and transforming 6Nomads data for predictive analytics. """
    def __init__(self):
        """ Initializer method for object instance creation. """
        self.REL_PATH_TO_INT_DATA_TRAIN = "../data/interim/train_i.csv"
        self.REL_PATH_TO_INT_DATA_TEST = "../data/interim/test_i.csv"
        
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
            return pd.read_csv(self.REL_PATH_TO_INT_DATA_TRAIN, index_col=0)
        
        # Independently load testing data
        if which == "test":
            return pd.read_csv(self.REL_PATH_TO_INT_DATA_TEST, index_col=0)
        else:
            df_train = pd.read_csv(self.REL_PATH_TO_INT_DATA_TRAIN, index_col=0)
            df_test = pd.read_csv(self.REL_PATH_TO_INT_DATA_TEST, index_col=0)
            
            # Load merged training and testing data
            if which == "all":
                return pd.concat([df_train, df_test], keys=["train", "test"], sort=True)
            
            # Load separated training and testing data (DEFAULT)
            if which == "both":
                return df_train, df_test
            
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
                - str(csv): Dataset is formatted as comma-separated values file architecture. (DEFAULT)
                - str(excel): Dataset is formatted as Excel spreadsheet file architecture.
                
        OUTPUTS:
            NoneType: Saving data to memory is performed outside context of object and does not return new object(s).
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
            