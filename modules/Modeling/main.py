# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 09:32:15 2020

@author: Dohyeon Lee, Jonghwan Park
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
from preprocessing import preprocessing as pre
from models import test_process as tp, vali_process as vp

parser = argparse.ArgumentParser(description='Process to set parameters.')
parser.add_argument('--estimators', type=int, default=25000, help = "Set Number of estimators of lightgbm model.")
parser.add_argument('--leaves', type=int, default=300, help = "Set main parameter to control the complexity of the tree model.")
parser.add_argument('--lr', type=float, default=0.01, help = "Set learning rate of leaf nodes for model.")
parser.add_argument('--topk', type=int, default=20, help = "Set number of candidates estimators for final prediction of model.")
args = parser.parse_args()

def learning_process_final(
        train_path, 
        test_path, 
        sample_path, 
        clf_code, 
        n_estimators,
        num_leaves, 
        learning_rate, 
        top_k
        ):
    
    """Returns a pd.DataFrame for test prediction.

    Parameters
    ----------
    train_path : str
        The path of the train dataset.
    test_path : str
        The path of the test dataset.
    sample_path : str
        The path of the sample csv.
    clf_mode : str
        The type of classifier for model 
    n_estimators : int
        The number of estimators for model
    num_leaves : int
        The number of leaf nodes for model
    learning_rate : float
        The Learning_rate of leaf nodes for model
    top_k : int
        The number of candidates estimators for final prediction of model
    """

    start = time.time()

    train_cls = pre(train_path, True)
    test_cls = pre(test_path, False)

    train_df = train_cls.main()
    test_df = test_cls.main()

    sample_df = pd.read_csv(sample_path)

    print('data_loading_time : %f.2'%(time.time() - start))

    start = time.time()
    main_cls = tp(train_df=train_df, 
                  test_df=test_df, 
                  sample_df=sample_df, 
                  clf_code=clf_code, 
                  n_estimators=(n_estimators,n_estimators), 
                  num_leaves=(num_leaves,num_leaves), 
                  lr=learning_rate,
                  top_k=top_k)

    result_sample = main_cls.main_experience()

    print('learning_time : %f.2'%(time.time() - start))

    return result_sample


def learning_process_vali(
        train_path, 
        clf_code, 
        n_estimators,
        num_leaves, 
        learning_rate, 
        top_k
        ):
    """Returns a pd.DataFrame for validation results.

    Parameters
    ----------
    train_path : str
        The path of the train dataset.
    clf_mode : str
        The type of classifier for model 
    n_estimators : int
        The number of estimators for model
    num_leaves : int
        The number of leaf nodes for model
    learning_rate : float
        The Learning_rate of leaf nodes for model
    top_k : int
        The number of candidates estimators for final prediction of model
    """
    
    start = time.time()
    
    train_cls = pre(train_path, True)
    train_df = train_cls.main()
        
    print('data_loading_time : %f.2'%(time.time() - start))
    
    #####
    start = time.time()

    main_cls = vp(base_df=train_df, 
                  clf_code=clf_code, 
                  n_estimators=n_estimators,
                  num_leaves=num_leaves,
                  lr=learning_rate, 
                  top_k=top_k)

    result_sample = main_cls.process_learning()
    
    print('learning_time : %f.2'%(time.time() - start))
    
    return result_sample

def open_dirs(dir_):
    """Make directory using path.

    Parameters
    ----------
    dir_ : str
        The path of the directory.
    """
    if isinstance(dir_, str):
        if not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)

    else:
        for single_dir in dir_:
            if not os.path.exists(dir_):
                os.makedirs(dir_, exist_ok=True)

def save_csv_s(df, root_dir, leaf_name, index=False, encoding='utf-8'):
    """Save csv file using path.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    root_dir : str
        The path of the directory.
    leaf_name : str
        The name of the csv file.
    index : boolean, default=False
        remain index of DataFrame for csv file.
    encoding : str, default='utf-8'
    """
    open_dirs(root_dir)

    df.to_csv(os.path.join(root_dir, leaf_name), index=index, encoding=encoding)

if __name__ == "__main__":

    train_path = r'../../resources/dataset/insurance/train.csv'
    test_path = r'../../resources/dataset/insurance/test.csv'
    sample_path = r'../../resources/dataset/insurance/sample.csv'

    # n_estimators=25000
    # num_leaves=300        
    # learning_rate = 0.01
    # top_k=20

    n_estimators=args.estimators
    num_leaves=args.leaves
    learning_rate = args.lr
    top_k=args.topk
    
    result_sample = learning_process_final(train_path, test_path, sample_path, ('lgbm',n_estimators, num_leaves,learning_rate,top_k))

    root_dir = r'../../results/submit/'
    leaf_name = 'final_submit.csv'
    save_csv_s(result_sample, root_dir, leaf_name)



    
    
