import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torcheval.metrics import R2Score
from keras.layers import Input, Dense, LSTM, Flatten, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention
import time
import os
import numpy as np
from sklearn import metrics


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
    def __init__(self, train_path=None,test_path=None,flag="train",num_T=1):
        mms = MinMaxScaler(feature_range=(0, 1))
        self.flag = flag
        # self.features=['ECG_Rate', 
        #                'EDA_Tonic',
        #                'EDA_Phasic',
        #                'EMG_Amplitude_coru',
        #                'EMG_Amplitude_trap',
        #                'EMG_Amplitude_zygo',
        #                'PPG_Rate',
        #                'Skt',
        #                # 'Video'
        #               ]

        self.features = [ 
            'Video', 
                         'EDA_Tonic', 
                         'EDA_Tonic_max_level_shift_21','EDA_Tonic_spike_21','EDA_Tonic_std1st_der_21', 
                        #  'EDA_Tonic_mean_21', 'EDA_Tonic_mean_lag60', 'EDA_Tonic_mean_lag120', 
                        #  'EDA_Tonic_mean_lag180', 'EDA_Tonic_mean_lag240', 'EDA_Tonic_mean_lag300',
                         'EDA_Phasic', 
                         'EDA_Phasic_max_level_shift_21','EDA_Phasic_spike_21', 'EDA_Phasic_std1st_der_21',
                         #'EDA_Phasic_mean_lag60','EDA_Phasic_mean_lag120', 'EDA_Phasic_mean_lag180', 'EDA_Phasic_mean_lag240', 'EDA_Phasic_mean_lag300',
                         'Skt', 
                         'Skt_max_level_shift_21','Skt_spike_21','Skt_std1st_der_21',
                        #  'Skt_mean_lag60', 'Skt_mean_lag120', 'Skt_mean_lag180', 'Skt_mean_lag240', 'Skt_mean_lag300', 
                         'ECG_Rate', 
                         'PPG_Rate', 
                         'RSP_Rate', 
                         'EMG_Amplitude_zygo', 
                         'EMG_Amplitude_coru', 
                         'EMG_Amplitude_trap',
        ]
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
            self.train_data = mms.fit_transform(self.train_df.loc[:,self.features].astype(np.float32).values)
            self.train_label = self.train_df.loc[:,self.labels].astype(np.float32).values
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
            # self.test_data = moving_average(self.test_data,100)
            # self.test_data = downsample(self.test_data,100)
            
            self.test_label = self.test_df.loc[:,self.labels].astype(np.float32).values
            # self.test_label = moving_average(self.test_label,100)
            # self.test_label = downsample(self.test_label,100)

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
        
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout_rate,num_T=20):
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

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test = Dataset_Epic(test_path="../ID_20_2hz_test_dynamic_features_lagstep60",flag="test",num_T=1)
    train = Dataset_Epic(train_path="../ID_80_2hz_train_dynamic_features_lagstep60",flag="train",num_T=1)
    train_dataloader = DataLoader(
        train,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )




    best_model = torch.load('./best_model_dropout_l2_nooverlap_80_ID_tanming_cross_val_selected_dyn_with_videoid_no_CuDNN.pt',weights_only=False)
    print(best_model)

    x_train = []
    for index, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        x_train.append(x)

    x_train = torch.cat(x_train, dim=0)

    epoch_loss = 0.0
    criterion = nn.MSELoss()
    predictions = []
    actuals = []
    metric = R2Score()
    x_test = []
    for index, (x, y) in enumerate(test_dataloader):
        x, y = x.to(device), y.to(device)
        outputs = best_model(x)
        predictions.extend(outputs.detach().cpu().numpy())
        actuals.extend(y.detach().cpu().numpy())
        loss = criterion(outputs, y)
        x_test.append(x)
        metric.update(outputs,y)

        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(test_dataloader)
    x_test = torch.cat(x_test,dim=0)
    r2 = metric.compute()  
    print(f"lost {avg_loss}")
    print(f"r2 {r2}")

    cpu = torch.device("cpu")
    best_model.to(cpu)
    x_train = x_train.to(cpu)[:20]
    x_test = x_test.to(cpu)[:5]
    features = [ 
                'Video', 
                             'EDA_Tonic', 
                             'EDA_Tonic_max_level_shift_21','EDA_Tonic_spike_21','EDA_Tonic_std1st_der_21', 
                            #  'EDA_Tonic_mean_21', 'EDA_Tonic_mean_lag60', 'EDA_Tonic_mean_lag120', 
                            #  'EDA_Tonic_mean_lag180', 'EDA_Tonic_mean_lag240', 'EDA_Tonic_mean_lag300',
                             'EDA_Phasic', 
                             'EDA_Phasic_max_level_shift_21','EDA_Phasic_spike_21', 'EDA_Phasic_std1st_der_21',
                             #'EDA_Phasic_mean_lag60','EDA_Phasic_mean_lag120', 'EDA_Phasic_mean_lag180', 'EDA_Phasic_mean_lag240', 'EDA_Phasic_mean_lag300',
                             'Skt', 
                             'Skt_max_level_shift_21','Skt_spike_21','Skt_std1st_der_21',
                            #  'Skt_mean_lag60', 'Skt_mean_lag120', 'Skt_mean_lag180', 'Skt_mean_lag240', 'Skt_mean_lag300', 
                             'ECG_Rate', 
                             'PPG_Rate', 
                             'RSP_Rate', 
                             'EMG_Amplitude_zygo', 
                             'EMG_Amplitude_coru', 
                             'EMG_Amplitude_trap',
            ]
    print(x_train.shape)
    explainer = shap.DeepExplainer(best_model, x_train)
    # explain the the testing instances (can use fewer instanaces)
    # explaining each prediction requires 2 * background dataset size runs
    shap_values = explainer.shap_values(x_test)
    # init the JS visualization code
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)

