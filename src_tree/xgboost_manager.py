import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
)
import numpy as np
import pandas as pd

class XgboostManager():
    def __init__(self, path,y, expanded_eda = False):
        self.path = path
        self.y = y
        self.expanded_eda = expanded_eda
        self.y_df = None
        self.X_df = None
        self.stats = {}
        

    def prepare_data(self):
        df = pd.read_csv(self.path)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        self.y_df = df[self.y]
        self.X_df = df.drop(self.y, axis=1)
    
    def train_model(self,test_size=0.2,n_estimators = 100, max_depth=5, learning_rate=0.1, scale_pos_weight=4.0,):
        X_train, X_test, y_train, y_test = train_test_split(self.X_df, self.y_df, test_size=test_size, random_state=42, stratify=self.y_df)
        if self.expanded_eda:
            X_train, X_test = self._exp_eda(X_train, X_test)
            cols_to_drop = ["customer_id", "credit_score","credit_card","tenure","age_group","estimated_salary"]
            X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
            X_test = X_test.drop(columns=cols_to_drop, errors='ignore')
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",        
            enable_categorical=True,   
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] 

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        print(f"\n--- Metryki dla XGBoost (expanded_eda={self.expanded_eda}) ---")
        for name, value in metrics.items():
            print(f"{name} : {value}")

        print("\n--- Macierz Pomyłek ---")
        print(confusion_matrix(y_test, y_pred))

        return metrics
    
    def _exp_eda(self, X_train, X_test):
        X_train = self._apply_feature_engineering(X_train, is_training=True)
        X_test = self._apply_feature_engineering(X_test, is_training=False)
        return X_train, X_test


    def _apply_feature_engineering(self, df, is_training=True):
        df = df.copy()
        
        if self.expanded_eda:
            features_to_scale = ['credit_score', 'balance', 'tenure']
            
            for col in features_to_scale:
                stat_name = f'med_{col}'
                
                if is_training:
                    self.stats[stat_name] = df.groupby("age_group")[col].median()
                
                df[f"s_{col}"] = df[col] / df["age_group"].map(self.stats[stat_name])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
            
        return df


            

    



