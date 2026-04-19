import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def clean_df_raw(df_raw):
    df_clean = pd.DataFrame(index = df_raw.index)
    df_clean['!age'] = df_raw['Age']/100
    df_clean[['!gender_f','!gender_m','!gender_x']] = pd.get_dummies(df_raw, columns=['Gender', ])[['Gender_Female','Gender_Male','Gender_Other']]
    df_clean['!height'] = df_raw['Height_cm'].apply(lambda x: (x-120)/100)
    df_clean['!weight'] = df_raw['Initial_Weight_kg'].apply(lambda x: sigmoid(0.08*(x-120)))
    df_clean['!stress'] = df_raw['Stress_Level']/10
    df_clean['!sleep'] = df_raw['Sleep_Hours']/12
    df_clean['!caffeine'] = df_raw['Caffeine_mg'].apply(lambda x: sigmoid(0.015*(x-260)))
    df_clean['!calories'] = df_raw['Calories_Consumed'].fillna(df_raw['Calories_Consumed'].mean())/4200
    df_clean.loc[df_clean['!calories']<0,['!calories']] = 0
    df_clean['!protein'] = df_raw['Protein_g']/305
    df_clean.loc[df_clean['!protein']<0,['!protein']] = 0
    df_clean['!carbs'] = df_raw['Carbs_g']/480
    df_clean.loc[df_clean['!carbs']<0,['!carbs']] = 0
    df_clean['!fat'] = df_raw['Fat_g']/160
    df_clean.loc[df_clean['!fat']<0,['!fat']] = 0
    df_clean['!steps'] = (df_raw['Steps']-7600)/2500
    df_clean[['!work_carb','!work_na','!work_strn','!work_yoga']] = pd.get_dummies(df_raw, columns=['Workout_Type', ])[
        ['Workout_Type_Cardio', 'Workout_Type_None', 'Workout_Type_Strength', 'Workout_Type_Yoga']
    ]
    df_clean['!workintensity'] = df_raw['Workout_Intensity']/10
    df_clean['!temp'] = df_raw['Temp_C'].apply(lambda x: sigmoid(-0.12*(x-20)))

    df_clean['#y#'] = df_raw['Weight_Change']
    return df_clean

class LWDataset(Dataset):
    
    def __init__(self, df):
        self.x = torch.tensor(df[df.columns[df.columns.str.contains('!')]].values,dtype=torch.float32)
        self.y = torch.tensor(df['#y#'].values,dtype=torch.float32).unsqueeze(dim=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class MLP(nn.Module):
 
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 48)
        self.fc2 = nn.Linear(48, 12)
        self.fc3 = nn.Linear(12, 3)
        self.fc4 = nn.Linear(3, 1)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y_pred = self.fc4(x)
        return y_pred