import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data import Dataset, DataLoader,Subset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GroupKFold

import time
import os
import numpy as np
from sklearn import metrics
from torcheval.metrics import R2Score


#LossesbyFold['FlexEmbedScatter'] = trials_flex_scatterEmbed.trials[trials_flex_scatterEmbed.best_trial['tid']]['result']['fold_losses']
#LossesbyFold['lastLossesFlexEmbedScatter'] = trials_flex_scatterEmbed.trials[-1]['result']['fold_losses']

def moving_average(data, window_size):
    if data.ndim == 1:
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='same')
    elif data.ndim == 2:
        smoothed_data = np.zeros_like(data)
        window = np.ones(window_size) / window_size
        for i in range(data.shape[1]):
            smoothed_data[:, i] = np.convolve(data[:, i], window, mode='same')
        return smoothed_data

def downsample(data, factor):
    if data.ndim == 1:
        return data[::factor]
    elif data.ndim == 2:
        return data[::factor, :]


class Dataset_Epic(Dataset):
    def __init__(self, train_path=None,test_path=None,flag="train",num_T=1,downsample_rate=10):
        mms = MinMaxScaler(feature_range=(0, 1))
        self.flag = flag
        self.features=['ECG_Rate', 
                       'EDA_Tonic',
                       'EDA_Phasic',
                       'EMG_Amplitude_coru',
                       'EMG_Amplitude_trap',
                       'EMG_Amplitude_zygo',
                       'PPG_Rate',
                       'Skt',
                    #    'Video',
                      ]
        # self.features=['ECG_Rate', 
        #                'ECG_Rate_entropy_401', 'ECG_Rate_max_level_shift_401', 'ECG_Rate_max_var_shift_401', 
        #                'ECG_Rate_max_kl_shift_401', 'ECG_Rate_hurst_401', 'ECG_Rate_spike_401', 'ECG_Rate_arch_acf_401', 
        #                'ECG_Rate_garch_acf_401', 'ECG_Rate_ARCH.LM_401', 'ECG_Rate_std1st_der_401', 'ECG_Rate_mean_401', 
        #                'ECG_Rate_mean_lag20', 'ECG_Rate_mean_lag40', 'ECG_Rate_mean_lag60', 'ECG_Rate_mean_lag80', 
        #                'ECG_Rate_mean_lag100', 'ECG_Rate_mean_lag120', 'ECG_Rate_mean_lag140', 'ECG_Rate_mean_lag160', 
        #                'ECG_Rate_mean_lag180', 'ECG_Rate_mean_lag200', 
        #                'EDA_Phasic', 
        #                'EDA_Phasic_entropy_401', 
        #                'EDA_Phasic_max_level_shift_401', 'EDA_Phasic_max_var_shift_401', 'EDA_Phasic_max_kl_shift_401', 
        #                'EDA_Phasic_hurst_401', 'EDA_Phasic_spike_401', 'EDA_Phasic_arch_acf_401', 'EDA_Phasic_garch_acf_401', 
        #                'EDA_Phasic_ARCH.LM_401', 'EDA_Phasic_std1st_der_401', 'EDA_Phasic_mean_401', 'EDA_Phasic_mean_lag20', 
        #                'EDA_Phasic_mean_lag40', 'EDA_Phasic_mean_lag60', 'EDA_Phasic_mean_lag80', 'EDA_Phasic_mean_lag100', 
        #                'EDA_Phasic_mean_lag120', 'EDA_Phasic_mean_lag140', 'EDA_Phasic_mean_lag160', 'EDA_Phasic_mean_lag180', 
        #                'EDA_Phasic_mean_lag200', 
        #                'EDA_Tonic',
        #                'EDA_Tonic_entropy_401', 'EDA_Tonic_max_level_shift_401', 
        #                'EDA_Tonic_max_var_shift_401', 'EDA_Tonic_max_kl_shift_401', 'EDA_Tonic_hurst_401', 
        #                'EDA_Tonic_spike_401', 'EDA_Tonic_arch_acf_401', 'EDA_Tonic_garch_acf_401', 'EDA_Tonic_ARCH.LM_401', 
        #                'EDA_Tonic_std1st_der_401', 'EDA_Tonic_mean_401', 'EDA_Tonic_mean_lag20', 'EDA_Tonic_mean_lag40', 
        #                'EDA_Tonic_mean_lag60', 'EDA_Tonic_mean_lag80', 'EDA_Tonic_mean_lag100', 'EDA_Tonic_mean_lag120', 
        #                'EDA_Tonic_mean_lag140', 'EDA_Tonic_mean_lag160', 'EDA_Tonic_mean_lag180', 'EDA_Tonic_mean_lag200', 
        #                'EMG_Amplitude_coru',
        #                'EMG_Amplitude_coru_entropy_401', 'EMG_Amplitude_coru_max_level_shift_401', 
        #                'EMG_Amplitude_coru_max_var_shift_401', 'EMG_Amplitude_coru_max_kl_shift_401', 
        #                'EMG_Amplitude_coru_hurst_401', 'EMG_Amplitude_coru_spike_401', 'EMG_Amplitude_coru_arch_acf_401', 
        #                'EMG_Amplitude_coru_garch_acf_401', 'EMG_Amplitude_coru_ARCH.LM_401', 
        #                'EMG_Amplitude_coru_std1st_der_401', 'EMG_Amplitude_coru_mean_401', 'EMG_Amplitude_coru_mean_lag20', 
        #                'EMG_Amplitude_coru_mean_lag40', 'EMG_Amplitude_coru_mean_lag60', 'EMG_Amplitude_coru_mean_lag80', 
        #                'EMG_Amplitude_coru_mean_lag100', 'EMG_Amplitude_coru_mean_lag120', 'EMG_Amplitude_coru_mean_lag140', 
        #                'EMG_Amplitude_coru_mean_lag160', 'EMG_Amplitude_coru_mean_lag180', 'EMG_Amplitude_coru_mean_lag200', 
        #                'EMG_Amplitude_trap', 
        #                'EMG_Amplitude_trap_entropy_401', 'EMG_Amplitude_trap_max_level_shift_401', 
        #                'EMG_Amplitude_trap_max_var_shift_401', 'EMG_Amplitude_trap_max_kl_shift_401', 
        #                'EMG_Amplitude_trap_hurst_401', 'EMG_Amplitude_trap_spike_401', 'EMG_Amplitude_trap_arch_acf_401', 
        #                'EMG_Amplitude_trap_garch_acf_401', 'EMG_Amplitude_trap_ARCH.LM_401', 
        #                'EMG_Amplitude_trap_std1st_der_401', 'EMG_Amplitude_trap_mean_401', 'EMG_Amplitude_trap_mean_lag20', 
        #                'EMG_Amplitude_trap_mean_lag40', 'EMG_Amplitude_trap_mean_lag60', 'EMG_Amplitude_trap_mean_lag80', 
        #                'EMG_Amplitude_trap_mean_lag100', 'EMG_Amplitude_trap_mean_lag120', 'EMG_Amplitude_trap_mean_lag140', 
        #                'EMG_Amplitude_trap_mean_lag160', 'EMG_Amplitude_trap_mean_lag180', 'EMG_Amplitude_trap_mean_lag200', 
        #                'EMG_Amplitude_zygo', 
        #                'EMG_Amplitude_zygo_entropy_401', 'EMG_Amplitude_zygo_max_level_shift_401', 
        #                'EMG_Amplitude_zygo_max_var_shift_401', 'EMG_Amplitude_zygo_max_kl_shift_401', 
        #                'EMG_Amplitude_zygo_hurst_401', 'EMG_Amplitude_zygo_spike_401', 'EMG_Amplitude_zygo_arch_acf_401', 
        #                'EMG_Amplitude_zygo_garch_acf_401', 'EMG_Amplitude_zygo_ARCH.LM_401', 
        #                'EMG_Amplitude_zygo_std1st_der_401', 'EMG_Amplitude_zygo_mean_401', 'EMG_Amplitude_zygo_mean_lag20', 
        #                'EMG_Amplitude_zygo_mean_lag40', 'EMG_Amplitude_zygo_mean_lag60', 'EMG_Amplitude_zygo_mean_lag80', 
        #                'EMG_Amplitude_zygo_mean_lag100', 'EMG_Amplitude_zygo_mean_lag120', 'EMG_Amplitude_zygo_mean_lag140', 
        #                'EMG_Amplitude_zygo_mean_lag160', 'EMG_Amplitude_zygo_mean_lag180', 'EMG_Amplitude_zygo_mean_lag200', 
        #                'PPG_Rate', 
        #                'PPG_Rate_entropy_401', 'PPG_Rate_max_level_shift_401', 'PPG_Rate_max_var_shift_401', 
        #                'PPG_Rate_max_kl_shift_401', 'PPG_Rate_hurst_401', 'PPG_Rate_spike_401', 'PPG_Rate_arch_acf_401', 
        #                'PPG_Rate_garch_acf_401', 'PPG_Rate_ARCH.LM_401', 'PPG_Rate_std1st_der_401', 'PPG_Rate_mean_401', 
        #                'PPG_Rate_mean_lag20', 'PPG_Rate_mean_lag40', 'PPG_Rate_mean_lag60', 'PPG_Rate_mean_lag80', 
        #                'PPG_Rate_mean_lag100', 'PPG_Rate_mean_lag120', 'PPG_Rate_mean_lag140', 'PPG_Rate_mean_lag160', 
        #                'PPG_Rate_mean_lag180', 'PPG_Rate_mean_lag200', 
        #                'Skt', 
        #                'Skt_entropy_401', 'Skt_max_level_shift_401', 
        #                'Skt_max_var_shift_401', 'Skt_max_kl_shift_401', 'Skt_hurst_401', 'Skt_spike_401', 'Skt_arch_acf_401', 
        #                'Skt_garch_acf_401', 'Skt_ARCH.LM_401', 'Skt_std1st_der_401', 'Skt_mean_401', 'Skt_mean_lag20', 
        #                'Skt_mean_lag40', 'Skt_mean_lag60', 'Skt_mean_lag80', 'Skt_mean_lag100', 'Skt_mean_lag120', 
        #                'Skt_mean_lag140', 'Skt_mean_lag160', 'Skt_mean_lag180', 'Skt_mean_lag200'
        #               ]
        #EDA, skin temp, (max_level_shift, spike, std1st_der)
        # self.features = ['EDA_Tonic',
        #                 #  'EDA_Tonic_max_level_shift_21',
        #                  'EDA_Tonic_entropy_21', 'EDA_Tonic_max_level_shift_21', 'EDA_Tonic_max_var_shift_21', 'EDA_Tonic_max_kl_shift_21', 
        #                  'EDA_Tonic_hurst_21', 'EDA_Tonic_spike_21', 'EDA_Tonic_arch_acf_21', 'EDA_Tonic_garch_acf_21', 'EDA_Tonic_ARCH.LM_21', 
        #                  'EDA_Tonic_std1st_der_21', 'EDA_Tonic_mean_21', 'EDA_Tonic_mean_lag10', 'EDA_Tonic_mean_lag20', 'EDA_Tonic_mean_lag30', 
        #                  'EDA_Tonic_mean_lag40', 'EDA_Tonic_mean_lag50', 
        #                  'EDA_Phasic', 
        #                 #  'EDA_Phasic_max_level_shift_21',
        #                  'EDA_Phasic_entropy_21', 'EDA_Phasic_max_level_shift_21', 'EDA_Phasic_max_var_shift_21', 'EDA_Phasic_max_kl_shift_21', 
        #                  'EDA_Phasic_hurst_21', 'EDA_Phasic_spike_21', 'EDA_Phasic_arch_acf_21', 'EDA_Phasic_garch_acf_21', 'EDA_Phasic_ARCH.LM_21', 
        #                  'EDA_Phasic_std1st_der_21', 'EDA_Phasic_mean_21', 'EDA_Phasic_mean_lag10', 'EDA_Phasic_mean_lag20', 'EDA_Phasic_mean_lag30',
        #                  'EDA_Phasic_mean_lag40', 'EDA_Phasic_mean_lag50', 
                         
        #                  'ECG_Rate', 
        #                  'ECG_Rate_entropy_21', 'ECG_Rate_max_level_shift_21', 'ECG_Rate_max_var_shift_21', 'ECG_Rate_max_kl_shift_21', 
        #                  'ECG_Rate_hurst_21', 'ECG_Rate_spike_21', 'ECG_Rate_arch_acf_21', 'ECG_Rate_garch_acf_21', 'ECG_Rate_ARCH.LM_21', 
        #                  'ECG_Rate_std1st_der_21', 'ECG_Rate_mean_21', 'ECG_Rate_mean_lag10', 'ECG_Rate_mean_lag20', 'ECG_Rate_mean_lag30', 
        #                  'ECG_Rate_mean_lag40', 'ECG_Rate_mean_lag50', 
        #                  'PPG_Rate', 
        #                  'PPG_Rate_entropy_21', 'PPG_Rate_max_level_shift_21', 'PPG_Rate_max_var_shift_21', 'PPG_Rate_max_kl_shift_21', 'PPG_Rate_hurst_21', 
        #                  'PPG_Rate_spike_21', 'PPG_Rate_arch_acf_21', 'PPG_Rate_garch_acf_21', 'PPG_Rate_ARCH.LM_21', 'PPG_Rate_std1st_der_21', 'PPG_Rate_mean_21', 
        #                  'PPG_Rate_mean_lag10', 'PPG_Rate_mean_lag20', 'PPG_Rate_mean_lag30', 'PPG_Rate_mean_lag40', 'PPG_Rate_mean_lag50', 
        #                  'RSP_Rate',
        #                  'RSP_Rate_entropy_21', 'RSP_Rate_max_level_shift_21', 'RSP_Rate_max_var_shift_21', 'RSP_Rate_max_kl_shift_21', 'RSP_Rate_hurst_21', 
        #                  'RSP_Rate_spike_21', 'RSP_Rate_arch_acf_21', 'RSP_Rate_garch_acf_21', 'RSP_Rate_ARCH.LM_21', 'RSP_Rate_std1st_der_21', 'RSP_Rate_mean_21', 
        #                  'RSP_Rate_mean_lag10', 'RSP_Rate_mean_lag20', 'RSP_Rate_mean_lag30', 'RSP_Rate_mean_lag40', 'RSP_Rate_mean_lag50', 
        #                  'EMG_Amplitude_zygo',
        #                  'EMG_Amplitude_zygo_entropy_21', 'EMG_Amplitude_zygo_max_level_shift_21', 'EMG_Amplitude_zygo_max_var_shift_21', 'EMG_Amplitude_zygo_max_kl_shift_21',
        #                  'EMG_Amplitude_zygo_hurst_21', 'EMG_Amplitude_zygo_spike_21', 'EMG_Amplitude_zygo_arch_acf_21', 'EMG_Amplitude_zygo_garch_acf_21',
        #                  'EMG_Amplitude_zygo_ARCH.LM_21', 'EMG_Amplitude_zygo_std1st_der_21', 'EMG_Amplitude_zygo_mean_21', 'EMG_Amplitude_zygo_mean_lag10', 
        #                  'EMG_Amplitude_zygo_mean_lag20', 'EMG_Amplitude_zygo_mean_lag30', 'EMG_Amplitude_zygo_mean_lag40', 'EMG_Amplitude_zygo_mean_lag50', 
        #                  'EMG_Amplitude_coru',
        #                  'EMG_Amplitude_coru_entropy_21', 'EMG_Amplitude_coru_max_level_shift_21', 'EMG_Amplitude_coru_max_var_shift_21', 'EMG_Amplitude_coru_max_kl_shift_21', 
        #                  'EMG_Amplitude_coru_hurst_21', 'EMG_Amplitude_coru_spike_21', 'EMG_Amplitude_coru_arch_acf_21', 'EMG_Amplitude_coru_garch_acf_21', 
        #                  'EMG_Amplitude_coru_ARCH.LM_21', 'EMG_Amplitude_coru_std1st_der_21', 'EMG_Amplitude_coru_mean_21', 'EMG_Amplitude_coru_mean_lag10', 
        #                  'EMG_Amplitude_coru_mean_lag20', 'EMG_Amplitude_coru_mean_lag30', 'EMG_Amplitude_coru_mean_lag40', 'EMG_Amplitude_coru_mean_lag50', 
        #                  'EMG_Amplitude_trap',
        #                  'EMG_Amplitude_trap_entropy_21', 'EMG_Amplitude_trap_max_level_shift_21', 'EMG_Amplitude_trap_max_var_shift_21', 'EMG_Amplitude_trap_max_kl_shift_21', 
        #                  'EMG_Amplitude_trap_hurst_21', 'EMG_Amplitude_trap_spike_21', 'EMG_Amplitude_trap_arch_acf_21', 'EMG_Amplitude_trap_garch_acf_21', 'EMG_Amplitude_trap_ARCH.LM_21', 
        #                  'EMG_Amplitude_trap_std1st_der_21', 'EMG_Amplitude_trap_mean_21', 'EMG_Amplitude_trap_mean_lag10', 'EMG_Amplitude_trap_mean_lag20', 'EMG_Amplitude_trap_mean_lag30', 
        #                  'EMG_Amplitude_trap_mean_lag40', 'EMG_Amplitude_trap_mean_lag50', 
        #                  'Skt',
        #                 #  'Skt_max_level_shift_21',
        #                  'Skt_entropy_21', 'Skt_max_level_shift_21', 'Skt_max_var_shift_21', 'Skt_max_kl_shift_21', 'Skt_hurst_21', 'Skt_spike_21', 'Skt_arch_acf_21', 
        #                  'Skt_garch_acf_21', 'Skt_ARCH.LM_21', 'Skt_std1st_der_21', 'Skt_mean_21', 'Skt_mean_lag10', 'Skt_mean_lag20', 'Skt_mean_lag30', 'Skt_mean_lag40', 'Skt_mean_lag50',
        #                  'Video'
        #                  ]

        # self.features = ['Video', 
        #                  'EDA_Tonic', 
        #                  'EDA_Tonic_max_level_shift_21','EDA_Tonic_spike_21','EDA_Tonic_std1st_der_21', 
        #                 #  'EDA_Tonic_mean_21', 'EDA_Tonic_mean_lag60', 'EDA_Tonic_mean_lag120', 
        #                 #  'EDA_Tonic_mean_lag180', 'EDA_Tonic_mean_lag240', 'EDA_Tonic_mean_lag300',
        #                  'EDA_Phasic', 
        #                  'EDA_Phasic_max_level_shift_21','EDA_Phasic_spike_21', 'EDA_Phasic_std1st_der_21',
        #                  #'EDA_Phasic_mean_lag60','EDA_Phasic_mean_lag120', 'EDA_Phasic_mean_lag180', 'EDA_Phasic_mean_lag240', 'EDA_Phasic_mean_lag300',
        #                  'Skt', 
        #                  'Skt_max_level_shift_21','Skt_spike_21','Skt_std1st_der_21',
        #                 #  'Skt_mean_lag60', 'Skt_mean_lag120', 'Skt_mean_lag180', 'Skt_mean_lag240', 'Skt_mean_lag300', 
        #                  'ECG_Rate', 
        #                  'PPG_Rate', 
        #                  'RSP_Rate', 
        #                  'EMG_Amplitude_zygo', 
        #                  'EMG_Amplitude_coru', 
        #                  'EMG_Amplitude_trap', 

        #                 #  'EDA_Tonic_entropy_21', 'EDA_Tonic_max_level_shift_21', 'EDA_Tonic_max_var_shift_21', 'EDA_Tonic_max_kl_shift_21', 'EDA_Tonic_hurst_21', 'EDA_Tonic_spike_21', 
        #                 #  'EDA_Tonic_arch_acf_21', 'EDA_Tonic_garch_acf_21', 'EDA_Tonic_ARCH.LM_21', 'EDA_Tonic_std1st_der_21', 'EDA_Tonic_mean_21', 'EDA_Tonic_mean_lag60', 'EDA_Tonic_mean_lag120', 
        #                 #  'EDA_Tonic_mean_lag180', 'EDA_Tonic_mean_lag240', 'EDA_Tonic_mean_lag300', 
        #                 #  'EDA_Phasic', 
        #                 #  'EDA_Phasic_entropy_21', 'EDA_Phasic_max_level_shift_21', 'EDA_Phasic_max_var_shift_21', 'EDA_Phasic_max_kl_shift_21', 'EDA_Phasic_hurst_21', 'EDA_Phasic_spike_21', 
        #                 #  'EDA_Phasic_arch_acf_21', 'EDA_Phasic_garch_acf_21', 'EDA_Phasic_ARCH.LM_21', 'EDA_Phasic_std1st_der_21', 'EDA_Phasic_mean_21', 'EDA_Phasic_mean_lag60',
        #                 #  'EDA_Phasic_mean_lag120', 'EDA_Phasic_mean_lag180', 'EDA_Phasic_mean_lag240', 'EDA_Phasic_mean_lag300', 
        #                 #  'ECG_Rate', 
        #                 #  'ECG_Rate_entropy_21', 'ECG_Rate_max_level_shift_21', 'ECG_Rate_max_var_shift_21', 'ECG_Rate_max_kl_shift_21', 'ECG_Rate_hurst_21', 'ECG_Rate_spike_21',
        #                 #  'ECG_Rate_arch_acf_21', 'ECG_Rate_garch_acf_21', 'ECG_Rate_ARCH.LM_21', 'ECG_Rate_std1st_der_21', 'ECG_Rate_mean_21', 'ECG_Rate_mean_lag60', 'ECG_Rate_mean_lag120', 
        #                 #  'ECG_Rate_mean_lag180', 'ECG_Rate_mean_lag240', 'ECG_Rate_mean_lag300', 
        #                 #  'PPG_Rate', 
        #                 #  'PPG_Rate_entropy_21', 'PPG_Rate_max_level_shift_21', 'PPG_Rate_max_var_shift_21', 'PPG_Rate_max_kl_shift_21', 'PPG_Rate_hurst_21', 'PPG_Rate_spike_21', 
        #                 #  'PPG_Rate_arch_acf_21', 'PPG_Rate_garch_acf_21', 'PPG_Rate_ARCH.LM_21', 'PPG_Rate_std1st_der_21', 'PPG_Rate_mean_21', 'PPG_Rate_mean_lag60', 'PPG_Rate_mean_lag120', 
        #                 #  'PPG_Rate_mean_lag180', 'PPG_Rate_mean_lag240', 'PPG_Rate_mean_lag300', 
        #                 #  'RSP_Rate', 
        #                 #  'RSP_Rate_entropy_21', 'RSP_Rate_max_level_shift_21', 'RSP_Rate_max_var_shift_21', 'RSP_Rate_max_kl_shift_21', 'RSP_Rate_hurst_21', 'RSP_Rate_spike_21', 
        #                 #  'RSP_Rate_arch_acf_21', 'RSP_Rate_garch_acf_21', 'RSP_Rate_ARCH.LM_21', 'RSP_Rate_std1st_der_21', 'RSP_Rate_mean_21', 'RSP_Rate_mean_lag60', 
        #                 #  'RSP_Rate_mean_lag120', 'RSP_Rate_mean_lag180', 'RSP_Rate_mean_lag240', 'RSP_Rate_mean_lag300', 
        #                 #  'EMG_Amplitude_zygo', 
        #                 #  'EMG_Amplitude_zygo_entropy_21', 'EMG_Amplitude_zygo_max_level_shift_21', 'EMG_Amplitude_zygo_max_var_shift_21', 'EMG_Amplitude_zygo_max_kl_shift_21', 
        #                 #  'EMG_Amplitude_zygo_hurst_21', 'EMG_Amplitude_zygo_spike_21', 'EMG_Amplitude_zygo_arch_acf_21', 'EMG_Amplitude_zygo_garch_acf_21', 'EMG_Amplitude_zygo_ARCH.LM_21', 
        #                 #  'EMG_Amplitude_zygo_std1st_der_21', 'EMG_Amplitude_zygo_mean_21', 'EMG_Amplitude_zygo_mean_lag60', 'EMG_Amplitude_zygo_mean_lag120', 'EMG_Amplitude_zygo_mean_lag180', 
        #                 #  'EMG_Amplitude_zygo_mean_lag240', 'EMG_Amplitude_zygo_mean_lag300', 
        #                 #  'EMG_Amplitude_coru', 
        #                 #  'EMG_Amplitude_coru_entropy_21', 'EMG_Amplitude_coru_max_level_shift_21', 'EMG_Amplitude_coru_max_var_shift_21', 'EMG_Amplitude_coru_max_kl_shift_21', 
        #                 #  'EMG_Amplitude_coru_hurst_21', 'EMG_Amplitude_coru_spike_21', 'EMG_Amplitude_coru_arch_acf_21', 'EMG_Amplitude_coru_garch_acf_21', 'EMG_Amplitude_coru_ARCH.LM_21', 
        #                 #  'EMG_Amplitude_coru_std1st_der_21', 'EMG_Amplitude_coru_mean_21', 'EMG_Amplitude_coru_mean_lag60', 'EMG_Amplitude_coru_mean_lag120', 'EMG_Amplitude_coru_mean_lag180', 
        #                 #  'EMG_Amplitude_coru_mean_lag240', 'EMG_Amplitude_coru_mean_lag300', 
        #                 #  'EMG_Amplitude_trap', 
        #                 #  'EMG_Amplitude_trap_entropy_21', 'EMG_Amplitude_trap_max_level_shift_21', 'EMG_Amplitude_trap_max_var_shift_21', 'EMG_Amplitude_trap_max_kl_shift_21', 
        #                 #  'EMG_Amplitude_trap_hurst_21', 'EMG_Amplitude_trap_spike_21', 'EMG_Amplitude_trap_arch_acf_21', 'EMG_Amplitude_trap_garch_acf_21', 'EMG_Amplitude_trap_ARCH.LM_21', 
        #                 #  'EMG_Amplitude_trap_std1st_der_21', 'EMG_Amplitude_trap_mean_21', 'EMG_Amplitude_trap_mean_lag60', 'EMG_Amplitude_trap_mean_lag120', 'EMG_Amplitude_trap_mean_lag180', 
        #                 #  'EMG_Amplitude_trap_mean_lag240', 'EMG_Amplitude_trap_mean_lag300', 
        #                 #  'Skt', 
        #                 #  'Skt_entropy_21', 'Skt_max_level_shift_21', 'Skt_max_var_shift_21', 'Skt_max_kl_shift_21', 'Skt_hurst_21', 'Skt_spike_21', 'Skt_arch_acf_21', 'Skt_garch_acf_21', 
        #                 #  'Skt_ARCH.LM_21', 'Skt_std1st_der_21', 'Skt_mean_21', 'Skt_mean_lag60', 'Skt_mean_lag120', 'Skt_mean_lag180', 'Skt_mean_lag240', 'Skt_mean_lag300'
        #                  ]
        self.labels=["valence","arousal"]
        
        self.train_df = pd.DataFrame()
        self.num_T = num_T
        self.test_df = pd.DataFrame()
        if self.flag == 'train':
            if not train_path.endswith("/"):
                train_path += "/"
            if os.path.isfile(train_path):
                self.train_df = pd.read_csv(train_path)
            
            if os.path.isdir(train_path):
                for (dirpath, dirnames, filenames) in os.walk(train_path):
                    for filename in filenames:
                        self.train_df = pd.concat([self.train_df,pd.read_csv(f"{dirpath}{filename}")],ignore_index=True)
            
            self.train_df['groups'] = self.train_df['ID'].astype(str)
            self.groups = self.train_df['groups'].values
            # self.groups = downsample(self.groups, downsample_rate)

            # print(self.train_df.columns.to_list())

            self.train_data = mms.fit_transform(self.train_df.loc[:,self.features].astype(np.float32).values)
            # self.train_data = moving_average(self.train_data, downsample_rate)
            # self.train_data = downsample(self.train_data, downsample_rate)

            self.train_label = self.train_df.loc[:,self.labels].astype(np.float32).values
            # self.train_label = moving_average(self.train_label, downsample_rate)
            # self.train_label = downsample(self.train_label, downsample_rate)
        if self.flag == 'test':
            
            if not test_path.endswith("/"):
                test_path += "/"

            if os.path.isfile(test_path):
                self.test_df = pd.read_csv(test_path)

            if os.path.isdir(test_path):
                for (dirpath, dirnames, filenames) in os.walk(test_path):
                    for filename in filenames:
                        self.test_df = pd.concat([self.test_df,pd.read_csv(f"{dirpath}{filename}")[self.features+self.labels]],ignore_index=True)
                        
            
            self.test_data = mms.fit_transform(self.test_df.loc[:,self.features].astype(np.float32).values)
            # self.test_data = moving_average(self.test_data, downsample_rate)
            # self.test_data = downsample(self.test_data, downsample_rate)

            self.test_label = self.test_df.loc[:,self.labels].astype(np.float32).values
            # self.test_label = moving_average(self.test_label, downsample_rate)
            # self.test_label = downsample(self.test_label, downsample_rate)
    def setNumT(self,num_T):
        self.num_T = num_T

    def __getitem__(self, index):
        x = None
        y = None
        if self.flag == 'train':
            x = self.train_data[index:index + self.num_T]
            y = self.train_label[index+self.num_T-1]
        elif self.flag == 'test':
            x = self.test_data[index:index + self.num_T]
            y = self.test_label[index+self.num_T-1]
        
        return x, y

    def __len__(self):
        # minus the label length
         # minus the label length
        if self.flag == 'train':
            return len(self.train_data) - self.num_T
        elif self.flag == 'test':
            return len(self.test_data) - self.num_T
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
train = Dataset_Epic(train_path="../ID_80_2hz_train_dynamic_features_lagstep60",flag="train",num_T=3)
test = Dataset_Epic(test_path="../ID_20_2hz_test_dynamic_features_lagstep60",flag="test",num_T=3)




# train_dataloader = DataLoader(
#     train,
#     batch_size=8,
#     shuffle=True,
#     num_workers=0,
#     drop_last=True,
#     pin_memory=True,
# )

import torch.optim as optim
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope

# Define an LSTM-based neural network for regression
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout_rate,num_T):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout_rate)
        #self.fc = nn.Linear(hidden_size, 2)  # Output a single value for each time step
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*num_T, 2),
        )

    def forward(self, x):
        # x is expected to have shape (batch_size, numFeatures, num_T)
        # Permute x to (batch_size, num_T, numFeatures) for LSTM
        
        #x = x.permute(0, 2, 1)
        batch_size, num_T, num_features = x.shape
        

        # Initialize hidden state and cell state on the same device as input x
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass the outputs through the fully connected layer to predict a value for each time step
        out = out.reshape(batch_size,-1)
        out = self.fc(out)  # out has shape (batch_size, num_T, 1)
        
        # Remove the last dimension, resulting in (batch_size, num_T)
        out = out.squeeze(-1)

        return out


# Define the objective function for LSTM regression
def objective_lstm(params, train, test,num_Features):
    epochs=5
    num_T = params['num_T']
    
    # Define model, loss function, and optimizer
   
    global best_loss, best_model, overall_bestr2
    
    # Forward pass with input tensor on the same device as the model
#     output = model(X_train)

    # Output shape should match y_train shape
#     print(f"Output shape: {output.shape}")  # Should be (num_Samples in set, num_T)
#     print(f"Target shape: {y_train.shape}") # Should be (num_Samples in set, num_T)

    # Loss computation (ensuring the output and target have the same shape)
    criterion = nn.MSELoss()
#     loss = criterion(output, y_train)
#     print(f"Loss: {loss.item()}")
    

    curr_trail_best_loss = float('inf')
    train.setNumT(num_T)
    test.setNumT(num_T)

    gkf = GroupKFold(n_splits=4)
    test_dataloader = DataLoader(test,batch_size=16,shuffle=False,num_workers=0,drop_last=False,pin_memory=True)
    
    average_r2 = 0.0
    iter_avg_r2 = 0.0
    iter_avg_loss = 0.0
    iter_avg_test_loss = 0.0

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train.train_data,train.train_label,train.groups)):
        torch.cuda.empty_cache()
        model = LSTMNet(input_size=num_Features, hidden_size=params['hidden_size'], num_layers=params['num_layers'],dropout_rate=params['dropout'],num_T=num_T).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        groups_np = train.groups
        train_indices = [i for i in train_idx if i < len(train)-num_T]
        train_indices = [i for i in train_indices if (groups_np[i] == groups_np[i:i+num_T]).all()]

        val_indices = [i for i in val_idx if i < len(train)-num_T]
        val_indices = [i for i in val_indices if (groups_np[i] == groups_np[i:i+num_T]).all()]

        train_subset = Subset(train, train_indices)
        val_subset = Subset(train, val_indices)

        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True,drop_last=True,pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=True,drop_last=True,pin_memory=True)

        epoch_best_r2 = 0.0
        last_epoch_r2 = 0.0
        last_epoch_loss = 0.0

        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            train_loss = 0.0
            val_loss = 0.0
            model.train()
            r2 = None
            metric = R2Score()
            
            for data, labels in train_loader:
                    
                # Zero the gradients
                optimizer.zero_grad()
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                train_loss += loss
                l2_reg = 0.0
                for name, param in model.named_parameters():
                
                    if name == 'output_layer.weight':
                        l2_reg = torch.sum(torch.square(param))
                
                loss += params['l2']*l2_reg
                loss.backward()
                optimizer.step()
            train_loss = train_loss/len(train_loader)

            model.eval()
            
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                metric.update(outputs,labels)
                loss = criterion(outputs, labels)
                val_loss += loss
            val_loss = val_loss/len(val_loader)
            last_epoch_loss = val_loss
            r2 = metric.compute()
            last_epoch_r2 = r2
            if r2 > epoch_best_r2:
                epoch_best_r2 = r2.detach().item()
            if r2 > overall_bestr2:
                overall_bestr2 = r2
        
            for index, (x, y) in enumerate(test_dataloader):

                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)

                epoch_loss += loss
            avg_loss = epoch_loss / len(test_dataloader)
        
            if avg_loss < best_loss:
                best_loss = avg_loss
                print('best test:-------------------')
                print(params)
                print(best_loss)
                best_model = model
                torch.save(best_model,"best_model_dropout_l2_nooverlap_80_ID_tanming_cross_val_no_dyn_no_video_id.pt")
            if avg_loss < curr_trail_best_loss:
                curr_trail_best_loss = avg_loss
            print(f'Epoch [{epoch + 1}/{epochs}], Test_Loss: {avg_loss:.4f}, Train_Loss:{train_loss:.4f}, Val_Loss:{val_loss:.4f}, R2:{r2:.4f}')
            metric.reset()
        iter_avg_r2 += epoch_best_r2
        iter_avg_loss += last_epoch_loss
    iter_avg_r2 /=4
    iter_avg_loss/=4
    # print(f'iter_avg_r2:{iter_avg_r2:.4f}')
    # print(f'last_epoch_r2:{last_epoch_r2:.4f}')
    print(f'iter_avg_loss:{iter_avg_loss}')
    
    
    return {'loss': iter_avg_loss, 'status': STATUS_OK}

# Define the hyperparameter search space for LSTM regression
space_lstm = {
    'hidden_size': scope.int(hp.quniform('hidden_size', 30, 180, 2)),
    'num_layers': scope.int(hp.quniform('num_layers', 1, 12, 1)),
    'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-1)),
    'dropout': scope.float(hp.quniform('dropout', 0.0, 0.5, 0.05)),
    'l2': scope.float(hp.quniform('l2', 0.0, 0.5, 0.05)),
    'num_T': scope.int(hp.quniform('num_T',1,5,1))
}

best_loss = float('inf')
overall_bestr2 = 0.0
best_model = None



# Run the optimization for LSTM regression
trials_lstm = Trials()
best_lstm = fmin(fn=lambda params: objective_lstm(params, train, test, len(train.features)), 
                 space=space_lstm, algo=tpe.suggest, max_evals=50, trials=trials_lstm)

print(f'overall_bestr2:{overall_bestr2:.4f}')

print("Best hyperparameters found for LSTM regression: ", best_lstm)
