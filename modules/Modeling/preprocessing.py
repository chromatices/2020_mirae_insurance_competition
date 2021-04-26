# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 09:32:15 2020

@author: Dohyeon Lee, Jonghwan Park
"""


import numpy as np
import pandas as pd

class preprocessing(object):
    
    def __init__(self, data_path, is_train):
        
        self.data_path = data_path
        self.is_train = is_train
        
    def load_csv(self, full_path, index_col = None, usecols = None, dtype = None):
    
        data  = pd.read_csv(filepath_or_buffer= full_path, index_col = index_col, usecols = usecols, dtype = dtype)
    
        return data
    
    
    def make_base(self, data_path):
    
        data = self.load_csv(data_path)
    
        data = data.drop(columns = ['ID'])
        data = data.rename(columns = {'target' : 'label_3'})
    
        return data


    def arr_to_idx(self, arr, oper_idx_dict):
    
        if len(np.unique(arr)) == len(arr):
    
            idx = oper_idx_dict[tuple(arr)]
    
        elif len(np.unique(arr)) == 1:
    
            idx = 16
    
        elif len(np.unique(arr)) == 2:
    
            values, counts = np.unique(arr, return_counts=True)
    
            if counts[np.argmax(values)] == 2 :
    
                temp_arr = np.ones(len(arr))
                temp_arr[np.argmin(values)] = 0
    
                idx = oper_idx_dict[tuple(temp_arr)]
    
            elif counts[np.argmax(values)] == 1 :
    
                temp_arr = np.zeros(len(arr))
                temp_arr[np.argmax(values)] = 1
    
                idx = oper_idx_dict[tuple(temp_arr)]
                
        return idx
    
    def make_oper_idx_dict(self):
        a1,a2,a3,a4 = [0,1,2], [0,2,1], [0,0,1], [0,1,0]    
        a5,a6,a7,a8 = [0,1,1], [1,0,2], [2,0,1], [1,0,0]    
        a9,a10,a11,a12 = [0,0,1], [1,0,1], [2,1,0], [1,2,0]    
        a13,a14,a15,a16 = [1,0,0], [0,1,0], [1,1,0],[0,0,0]
    
        oper_list = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16]
        oper_list = [tuple(x) for x in oper_list]
        oper_idx_dict = dict(zip(oper_list,list(np.arange(1,len(oper_list)+1,1))))
        
        return oper_idx_dict
    
    def make_zero_idx_dict(self):
    
        a1,a2,a3,a4 = [0,0,0],[1,0,0],[0,1,0],[0,0,1]
    
        a5,a6,a7,a8 = [1,1,0],[0,1,1],[1,0,1],[1,1,1]
    
        zero_list = [a1, a2, a3, a4, a5, a6, a7, a8]
        zero_list = [tuple(x) for x in zero_list]
    
        zero_idx_dict = dict(zip(zero_list,list(np.arange(1,len(zero_list) + 1,1))))
        
        return zero_idx_dict
    
    def one_hot_to_idx(self, arr, zero_idx_dict):
    
        idx  = zero_idx_dict[tuple(arr)]
    
        return idx
    
    def remove_null(self, base_df, col_name, value):
        return base_df.loc[base_df[col_name] != value]
    
    
    def make_conti_arr(self):
    
        dsas_avg_conti = ['dsas_avg_diag_bilg_isamt_s','dsas_avg_surop_bilg_isamt_s','dsas_avg_hspz_bilg_isamt_s','dsas_avg_optt_bilg_isamt_s']
        hsp_avg_conti = ['hsp_avg_diag_bilg_isamt_s','hsp_avg_surop_bilg_isamt_s','hsp_avg_hspz_bilg_isamt_s','hsp_avg_optt_bilg_isamt_s']
        bilg_conti = ['bilg_isamt_s','surop_blcnt_s','hspz_blcnt_s','optt_blcnt_s']
    
        conti_arr = np.array([dsas_avg_conti, hsp_avg_conti, bilg_conti])
        
        return conti_arr
    
    def make_mdct_nur_dict(self):
        
        mdct_nur_key = [(1,0),(1,1),(2,0),(2,1),(3,0)]
        mdct_nur_val = [0,1,2,3,4]
        mdct_nur_dict = dict(zip(mdct_nur_key, mdct_nur_val))
        
        return mdct_nur_dict
    
    def make_zero_idx(self, col_idx, base_df, conti_arr, zero_idx_dict): 
        temp_arr = (base_df[conti_arr.T[col_idx]] == 0).values*1
        return np.apply_along_axis(self.one_hot_to_idx, 1, temp_arr, zero_idx_dict)
    
    def make_val_idx(self, col_idx, base_df, conti_arr, oper_idx_dict):

        temp_arr = (base_df[conti_arr.T[col_idx]] == 0).values*1

        return np.apply_along_axis(self.arr_to_idx, 1, temp_arr, oper_idx_dict)
    
    
    def main(self):
        
        base_df = self.make_base(self.data_path)
        oper_idx_dict = self.make_oper_idx_dict()
        zero_idx_dict = self.make_zero_idx_dict()
        conti_arr = self.make_conti_arr()
        
        new_base_df = self.remove_null(base_df, 'mdct_inu_rclss_dcd', 9).reset_index(drop=True)
        mdct_nur_dict = self.make_mdct_nur_dict()
        
        new_base_mdct_nur_mask = [mdct_nur_dict[x] for x in list(zip(new_base_df['mdct_inu_rclss_dcd'].values.astype(int),new_base_df['nur_hosp_yn'].values.astype(int)))]
        
        mdct_df = pd.DataFrame(data = new_base_mdct_nur_mask, columns = ['mdct_nur'])

        new_base_df = pd.concat([mdct_df, new_base_df],axis=1)
        
        zero_total_arr = np.vstack(tuple([*(self.make_zero_idx(x, new_base_df, conti_arr,zero_idx_dict) for x in range(4))]))
        
        val_total_arr = np.vstack(tuple([*(self.make_val_idx(x, new_base_df, conti_arr, oper_idx_dict) for x in range(4))]))
        
        idx_df = pd.DataFrame(data = np.concatenate([zero_total_arr.T.astype(int),val_total_arr.T.astype(int)],axis=1), columns = ['diag_0','surop_0','hspz_0','optt_0','diag_val','surop_val','hspz_val','optt_val'])
        final_df = pd.concat([idx_df,new_base_df],axis=1)
        
        if self.is_train:
        
            new_cols = ['ac_ctr_diff','hsp_avg_optt_bilg_isamt_s','hsp_avg_surop_bilg_isamt_s','ar_rclss_cd',
     'fds_cust_yn','inamt_nvcd','hsp_avg_diag_bilg_isamt_s','blrs_cd','dsas_ltwt_gcd',
     'dsas_avg_diag_bilg_isamt_s','dsas_acd_rst_dcd','base_ym','kcd_gcd','hsp_avg_hspz_bilg_isamt_s',
     'optt_blcnt_s','prm_nvcd','surop_blcnt_s','dsas_avg_optt_bilg_isamt_s','isrd_age_dcd',
     'hspz_blcnt_s','dsas_avg_surop_bilg_isamt_s','urlb_fc_yn','dsas_avg_hspz_bilg_isamt_s',
     'smrtg_5y_passed_yn','ac_rst_diff','bilg_isamt_s','diag_0','surop_0','hspz_0',
     'optt_0','diag_val','surop_val','hspz_val','optt_val','mdct_nur','label_3']
            
        else:
            new_cols = ['ac_ctr_diff','hsp_avg_optt_bilg_isamt_s','hsp_avg_surop_bilg_isamt_s','ar_rclss_cd',
     'fds_cust_yn','inamt_nvcd','hsp_avg_diag_bilg_isamt_s','blrs_cd','dsas_ltwt_gcd',
     'dsas_avg_diag_bilg_isamt_s','dsas_acd_rst_dcd','base_ym','kcd_gcd','hsp_avg_hspz_bilg_isamt_s',
     'optt_blcnt_s','prm_nvcd','surop_blcnt_s','dsas_avg_optt_bilg_isamt_s','isrd_age_dcd',
     'hspz_blcnt_s','dsas_avg_surop_bilg_isamt_s','urlb_fc_yn','dsas_avg_hspz_bilg_isamt_s',
     'smrtg_5y_passed_yn','ac_rst_diff','bilg_isamt_s','diag_0','surop_0','hspz_0',
     'optt_0','diag_val','surop_val','hspz_val','optt_val','mdct_nur']
            
            
        
        return final_df[new_cols]
        
        


