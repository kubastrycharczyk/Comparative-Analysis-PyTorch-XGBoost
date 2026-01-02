import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class DataManager():
    def __init__(self, path, y_col, batch_size, split_sizes=(0.7, 0.15, 0.15), expanded_eda=False):
        self.path = path
        self.y_col = y_col
        self.batch_size = batch_size
        self.split_sizes = split_sizes
        self.expanded_eda = expanded_eda
        self.stats = {}  
        
    def _apply_feature_engineering(self, df, is_training=True):
        df = df.copy()
        
        if self.expanded_eda:
            features_to_scale = ['credit_score', 'balance', 'tenure']
            
            for col in features_to_scale:
                stat_name = f'med_{col}'
                
                if is_training:
                    self.stats[stat_name] = df.groupby("age_group")[col].median()
                
           
                df[f"s_{col}"] = df[col] / df["age_group"].map(self.stats[stat_name])
            
            df = df.fillna(0).replace([np.inf, -np.inf], 0)
            
        return df

    def prepare_loaders(self):
        df = pd.read_csv(self.path)
        
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - self.split_sizes[0]), 
            random_state=42, 
            stratify=df[self.y_col] 
        )
        
        val_size_adj = self.split_sizes[1] / (self.split_sizes[1] + self.split_sizes[2])
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=(1 - val_size_adj), 
            random_state=42, 
            stratify=temp_df[self.y_col]
        )

        train_df = self._apply_feature_engineering(train_df, is_training=True)
        val_df = self._apply_feature_engineering(val_df, is_training=False)
        test_df = self._apply_feature_engineering(test_df, is_training=False)

        to_drop = [self.y_col]

        if self.expanded_eda:
                to_drop = [self.y_col, "customer_id", "credit_score","credit_card","tenure","age_group","estimated_salary"]


        def create_loader(curr_df, shuffle=False):
            y = torch.tensor(curr_df[self.y_col].values, dtype=torch.float32).unsqueeze(1)
            X = curr_df.drop(columns=[c for c in to_drop if c in curr_df.columns])
            X = pd.get_dummies(X, drop_first=True, dtype=int) 
            X_tensor = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
            self.input_size = X.shape[1]
            return DataLoader(TensorDataset(X_tensor, y), batch_size=self.batch_size, shuffle=shuffle)

        train_counts = train_df[self.y_col].value_counts()
        self.pos_weight = (train_counts[0] / train_counts[1]) * 1.5

        return create_loader(train_df, True), create_loader(val_df), create_loader(test_df)