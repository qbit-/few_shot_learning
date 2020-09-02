"""
Utility functions for datasets
"""
import numpy as np
import pandas as pd


def split_proportionally_by_class(data: pd.DataFrame, frac: float):
    """
    Splits data proportionally by each of it's classes in the field 'class'
    """
    df_train = data.groupby('class').apply(
        lambda x: x.sample(frac=frac))
    df_train.index = df_train.index.droplevel(0)
    
    df_val = data.drop(df_train.index)
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    return df_train, df_val


def get_class_freq(data, num_classes=None, only_classes=None):
    """
    Calculates relative frequencies by the field 'class'.
    
    Classes are assumed to be consequtive integer numbers.
    If the class is not encountered its weight is 0.
    """

    class_freq = data['class'].value_counts()
    if only_classes:
        class_freq = class_freq[only_classes]

    rel_freq = class_freq / class_freq.sum() 
    
    if not num_classes and not only_classes:
        num_classes = np.max(rel_freq.index) + 1
    
    fractions = pd.Series(np.zeros(num_classes), name='fractions')
    fractions.update(rel_freq)

    return fractions

def get_class_weights(data, num_classes=None, only_classes=None):
    """
    Calculates inverse frequencies of classes.
    
    Classes are assumed to be consequtive integer numbers.
    If the class is not encountered its weight is 0.
    """
    fractions = get_class_freq(data, num_classes=num_classes, only_classes=only_classes)
    weights = 1. / fractions
    
    return weights.replace([np.inf, -np.inf], 0)

def get_upsampling_counts(data, new_freq):
    """
    Calculates the number of upsampled items 
    for each class to match specified frequencies of classes
    """
    # use only the classes present in data - we can not upsample
    # classes with 0 examples
    freq_present = new_freq[data['class'].unique()]
    if len(freq_present) != len(new_freq):
        print('warning: not all classes are found in data')
    # renormalize
    freq = freq_present / freq_present.sum()
    
    # find upsampling from a linear system
    freq_v = freq.sort_index().to_numpy()
    A = np.eye(len(freq_v)) - np.tile(freq_v.reshape([-1, 1]), [1, len(freq_v)])
    counts = data['class'].value_counts().sort_index()
    b = counts - freq_v * counts.sum()
    
    upsample_by = np.linalg.solve(A, b).round().astype('int')
    upsample_by = pd.Series(upsample_by, index=freq.index, name='upsample_by')

    return upsample_by

def get_resampled_data(data, new_freq, scale_dataset_by=1.0):
    """
    Upsamples the data to match specified frequencies of classes
    scale_dataset_by defines how much (>1.0) or less (<1.0) examples
    new data set will have, defaults to same size. 
    """
    # use only the classes present in data - we can not upsample
    # classes with 0 examples
    freq_present = new_freq[data['class'].unique()]
    if len(freq_present) != len(new_freq):
        print('warning: not all classes are found in data')
    # renormalize
    freq = freq_present / freq_present.sum()
    
    # get old frequencies
    counts = data['class'].value_counts().sort_index()
    old_freq = counts / counts.sum()
    # upsampling ratios
    upsample_ratios = freq / old_freq * scale_dataset_by
    
    # perform upsampling
    new_data = data.copy()
    new_data['upsample_ratio'] = data.apply(lambda x: upsample_ratios[x['class']], axis=1)
    
    new_data = new_data.groupby('class').apply(
        lambda x: x.sample(frac=x['upsample_ratio'].iloc[0], replace=True))
    new_data.index = new_data.index.droplevel(0)
    new_data = new_data.drop('upsample_ratio', axis=1)
    
    # shuffle the new data. This is needed to overcome some bug with DataModule
    new_data = new_data.sample(frac=1).reset_index(drop=True)
    return new_data