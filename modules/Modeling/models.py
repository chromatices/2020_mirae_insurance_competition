# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 09:32:38 2020

@author: Dohyeon Lee, Jonghwan Park
"""
from lightgbm import LGBMClassifier
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score

class test_process(object):
    """Make model using train dataset, then predict class of test dataset.

    Parameters
    ----------
    train_df : pd.DataFrame
        train dataset
    test_df : pd.DataFrame
        test dataset
    sample_df : pd.DataFrame
        sample csv
    clf_mode : str
        The type of classifier for model 
    n_estimators : int
        The number of estimators for model
    num_leaves : int
        The number of leaf nodes for model
    lr : float
        The Learning_rate of leaf nodes for model
    top_k : int
        The number of candidates estimators for final prediction of model
    """
    def __init__(
            self, 
            train_df, 
            test_df, 
            sample_df, 
            clf_code, 
            n_estimators, 
            num_leaves,
            lr,
            top_k
            ):
        
        self.train_df = train_df
        self.test_df = test_df
        self.sample_df = sample_df
        self.clf_code = clf_code
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.lr = lr
        self.top_k = top_k

    def get_model_lgbm(self, x_arr, y_arr, n_estimators, num_leaves):
        """Get fitted light gbm model.

        Parameters
        ----------
        x_arr : np.ndarray
            Feature data.
        y_arr : np.ndarray
            Label data.
        n_estimators : int
            The number of estimators for lgbm.
        num_leaves : int
            The number of leaf nodes for lbgm.
        """

        temp_clf = LGBMClassifier(boosting_type='gbdt',
                            n_estimators=n_estimators,
                            random_state=0,
                            n_jobs=2,
                            learning_rate=self.lr,
                            top_k = self.top_k,
                            num_leaves=num_leaves)

        temp_clf.fit(x_arr, y_arr)

        return temp_clf

    def split_ltwt(self, cv_df, col_name, value):
        """Get splitted DataFrame based on variable(dsas_ltwt_gcd).

        Parameters
        ----------
        cv_df : pd.DataFrame
            The DataFrame to be spllited.
        col_name : str
            The name of column used for split.
        value : int
            The value of column used for split.
        """
        ltwt_1_df = cv_df.loc[cv_df[col_name] == value]
        ltwt_23_df = cv_df.loc[cv_df[col_name] != value]

        return ltwt_1_df, ltwt_23_df

    def split_x_y(self, base_df):
        """Split DataFrame to variable dataset(np.ndarray), 
        and label(np.ndarray).

        Parameters
        ----------
        base_df : pd.DataFrame
            The DataFrame to be spllited.
        """
        x_arr, y_arr = base_df[base_df.columns[:-1]].values, base_df[base_df.columns[-1]].values

        return x_arr, y_arr

    def main_experience(self):
        """Models and predicts

        Notes
        -----
        This method includes four steps
        
        step_1 : Split dataset based on dsas_ltwt_gcd[1 / 2&3]
        step_2 : Split DataFrame to numpy.ndarrays for modeling
        step_3 : Get model for each case[dsas_ltwt_gcd 1 / dsas_ltwt_gcd 2&3]
        step_4 : Predict and complete sample csv

        """
        start = time.time()

        #step_1 
        trn_ltwt_1_df, trn_ltwt_23_df = self.split_ltwt(cv_df=self.train_df, col_name='dsas_ltwt_gcd', value=1)
        tst_ltwt_1_df, tst_ltwt_23_df = self.split_ltwt(cv_df=self.test_df, col_name='dsas_ltwt_gcd', value=1)

        tst_ltwt_1_idx, tst_ltwt_23_idx = tst_ltwt_1_df.index.values, tst_ltwt_23_df.index.values

        #step_2 
        trn_ltwt_1_x_arr, trn_ltwt_1_y_arr = self.split_x_y(trn_ltwt_1_df)
        trn_ltwt_23_x_arr, trn_ltwt_23_y_arr = self.split_x_y(trn_ltwt_23_df)

        tst_ltwt_1_x_arr, tst_ltwt_23_x_arr = tst_ltwt_1_df.values, tst_ltwt_23_df.values

        #step_3
        if self.clf_code == 'lgbm':
            model_ltwt_1 = self.get_model_lgbm(trn_ltwt_1_x_arr, trn_ltwt_1_y_arr, self.n_estimators[0], self.num_leaves[0])
            model_ltwt_23 = self.get_model_lgbm(trn_ltwt_23_x_arr, trn_ltwt_23_y_arr,self.n_estimators[1], self.num_leaves[1])

        #step_4
        pred_ltwt_1 = model_ltwt_1.predict(tst_ltwt_1_x_arr)
        pred_ltwt_23 = model_ltwt_23.predict(tst_ltwt_23_x_arr)

        target_arr = np.ones(self.test_df.shape[0])
        target_arr[tst_ltwt_1_idx] = pred_ltwt_1
        target_arr[tst_ltwt_23_idx] = pred_ltwt_23

        sample_df = self.sample_df
        sample_df['target'] = target_arr.astype(int)

        print('fold_time : %f.2'%(time.time() - start))

        return sample_df


    
class vali_process(object):
    """Make model using train dataset, then predict class of validation dataset.

    Parameters
    ----------
    base_df : pd.DataFrame
        train dataset
    clf_mode : str
        The type of classifier for model 
    n_estimators : int
        The number of estimators for model
    num_leaves : int
        The number of leaf nodes for model
    lr : float
        The Learning_rate of leaf nodes for model
    top_k : int
        The number of candidates estimators for final prediction of model
    """

    def __init__(
            self, 
            base_df, 
            clf_code, 
            n_estimators, 
            lr, 
            num_leaves, 
            top_k
            ):
        
        self.base_df = base_df
        self.clf_code = clf_code
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.lr = lr
        self.top_k=top_k
        self.x_arr, self.y_arr = self.split_x_y(self.base_df)
        self.cols = self.base_df.columns
        self.y_arr = self.y_arr.astype(int)

    def get_model_lgbm(self, x_arr, y_arr,n_estimators, num_leaves):
        """Get fitted light gbm model.

        Parameters
        ----------
        x_arr : np.ndarray
            Feature data.
        y_arr : np.ndarray
            Label data.
        n_estimators : int
            The number of estimators for lgbm.
        num_leaves : int
            The number of leaf nodes for lbgm.
        """
        temp_clf = LGBMClassifier(boosting_type='gbdt',
                            n_estimators=n_estimators,
                            learning_rate=self.lr,
                            random_state=0,
                            num_leaves=num_leaves,
                            top_k=self.top_k,
                            n_jobs=-1)
        
        temp_clf.fit(x_arr, y_arr)

        return temp_clf


    def get_generator(self):
        """Get generator including indexs of train dataset and valid dataset.        
        """
        n_splits = 5
        random_state = 0
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

        return skf.split(self.x_arr, self.y_arr)

    def process_learning(self):
        """Get results of each validation dataset.
        """
    
        result_df = pd.DataFrame(columns=
                                 ['fold', 'ltwt_1_f1', 'ltwt_1_acc',
                                  'ltwt_23_f1', 'ltwt_23_acc']
                                 )
        
        i=0
        
        for idx, (trn,tst) in enumerate(self.get_generator()):
            output = self.fold_experience(idx,self.x_arr[trn], self.y_arr[trn], self.x_arr[tst], self.y_arr[tst])

            result_df.loc[i] = output
            
            i+=1
            
        return result_df

    def split_ltwt(self, cv_df, col_name, value):
        """Get splitted DataFrame based on variable(dsas_ltwt_gcd).

        Parameters
        ----------
        cv_df : pd.DataFrame
            The DataFrame to be spllited.
        col_name : str
            The name of column used for split.
        value : int
            The value of column used for split.
        """
        ltwt_1_df = cv_df.loc[cv_df[col_name] == value]
        ltwt_23_df = cv_df.loc[cv_df[col_name] != value]

        return ltwt_1_df, ltwt_23_df

    def split_x_y(self, base_df):
        """Split DataFrame to variable dataset(np.ndarray), and label(np.ndarray).

        Parameters
        ----------
        base_df : pd.DataFrame
            The DataFrame to be spllited.
        """
        x_arr, y_arr = base_df[base_df.columns[:-1]].values, base_df[base_df.columns[-1]].values

        return x_arr, y_arr

    def fold_experience(self, *args):
        """Experience for each validation dataset

        Parameters
        ----------
        idx : int
            The order of validation dataset.
        trn_arr : np.ndarray
            The input data for modeling.
        trn_y_arr : pd.DataFrame
            The input data for modeling.
        tst_arr : pd.DataFrame
            The input data for test.
        tst_y_arr : pd.DataFrame
            The input data for test.
            
        Notes
        -----
        This method includes four steps
        
        step_1 : Split dataset based on dsas_ltwt_gcd[1 / 2&3]
        step_2 : Split DataFrame to numpy.ndarrays for modeling
        step_3 : Get model for each case[dsas_ltwt_gcd 1 / dsas_ltwt_gcd 2&3]
        step_4 : Predict and get performance results

        """
        start = time.time()

        idx, trn_arr,trn_y_arr, tst_arr,tst_y_arr = args
        
        #step_1 
        col_idx = int(np.where(self.cols == 'dsas_ltwt_gcd')[0])

        trn_ltwt_1_idx = np.where(trn_arr[:,col_idx] == 1)[0]
        trn_ltwt_23_idx = np.where(trn_arr[:,col_idx] != 1)[0]

        tst_ltwt_1_idx = np.where(tst_arr[:,col_idx] == 1)[0]
        tst_ltwt_23_idx = np.where(tst_arr[:,col_idx] != 1)[0]
        
        #step_2 

        ##train dataset
        trn_ltwt_1_x_arr, trn_ltwt_1_y_arr = trn_arr[trn_ltwt_1_idx],trn_y_arr[trn_ltwt_1_idx]
        trn_ltwt_23_x_arr, trn_ltwt_23_y_arr = trn_arr[trn_ltwt_23_idx],trn_y_arr[trn_ltwt_23_idx]

        ##test dataset
        tst_ltwt_1_x_arr, tst_ltwt_1_y_arr = tst_arr[tst_ltwt_1_idx],tst_y_arr[tst_ltwt_1_idx]
        tst_ltwt_23_x_arr, tst_ltwt_23_y_arr = tst_arr[tst_ltwt_23_idx],tst_y_arr[tst_ltwt_23_idx]
        
        #step_3 
        if self.clf_code == 'lgbm':
            model_ltwt_1 = self.get_model_lgbm(trn_ltwt_1_x_arr, trn_ltwt_1_y_arr, self.n_estimators[0],self.num_leaves[0])
            model_ltwt_23 = self.get_model_lgbm(trn_ltwt_23_x_arr, trn_ltwt_23_y_arr, self.n_estimators[1],self.num_leaves[1])
            
        #step_4 
        pred_ltwt_1 = model_ltwt_1.predict(tst_ltwt_1_x_arr)
        pred_ltwt_23 = model_ltwt_23.predict(tst_ltwt_23_x_arr)

        ltwt_1_metric = self.fold_get_metric(tst_ltwt_1_y_arr,pred_ltwt_1)
        ltwt_23_metric = self.fold_get_metric(tst_ltwt_23_y_arr,pred_ltwt_23)
        
        if idx == 0:
            print('fold_time : %.2f'%(time.time() - start) )

        return [idx]+list(ltwt_1_metric) + list(ltwt_23_metric)

    def fold_get_metric(self, real_y, pred_y):
        """Get results using performance measure.

        Parameters
        ----------
        real_y : np.ndarray
            The target class to be predicted
            
        pred_y : np.ndarray
            The predicted target class to be evaluated
        """
        f1_out = f1_score(real_y, pred_y, average='macro')
        acc_out = accuracy_score(real_y, pred_y)

        return f1_out,acc_out
    
    
    
