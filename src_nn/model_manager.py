import model
import trainer
import torch.nn as nn
import torch
import torch.optim as optim
from data_manager import DataManager
from pathlib import Path

class ModelManager():
    def __init__(self,
                  path, 
                  y, 
                  batch_size, 
                  split_size = [0.8, 0.1, 0.1],
                  epochs = 50,
                  patience= 20,
                  drop_value = 0.2,
                  expanded_eda = False):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = patience
        self.manager = DataManager(path, y, batch_size, split_size, expanded_eda=expanded_eda)
        self.loaders = None
        self.inner_model = None
        self.model_trainer = None
        self.epochs = epochs
        self.drop_value = drop_value

    def preparation(self):
        self.loaders = self.manager.prepare_loaders()
        
        input_dim = self.manager.input_size 
        pos_weight_value = self.manager.pos_weight
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)


        self.inner_model = model.Classifier(input_dim, self.drop_value).to(self.device)
        self.model_trainer = trainer.Trainer(self.inner_model,
                                optim.Adam(self.inner_model.parameters(), lr=0.001),
                                self.loaders,
                                loss_fn,
                                self.epochs,
                                device=self.device
                                )
        
        

    def train(self, metrics =  ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']):
        if self.model_trainer is None:
            self.preparation()
        self.model_trainer.train(patience=self.patience)
        print(f"\n--- Metryki dla Neural Network (expanded_eda={self.manager.expanded_eda}) ---")
        results=self.model_trainer.tester(0.5, metrics)
        return results
        






