import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Version: 0.3


class Trainer():
    def __init__(self, model, opt , loader, loss_f = nn.BCELoss(), epochs = 10, device ="cpu" ):
        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f.to(self.device)
        self.opt = opt
        self.epochs = epochs
        self.avg_train_loss = 0.0
        self.avg_val_loss = 0.0
        self.train_loader = loader[0]
        self.val_loader = loader[1]
        self.test_loader = loader[2]
        self.best_val_loss = float("inf")

    def _training(self):
        self.model.train()
        total_loss = 0.0
        for batch_id, (X, y) in enumerate(self.train_loader): 
            X = X.to(self.device)
            y = y.to(self.device)
            self.opt.zero_grad() 
            prediction = self.model(X) 
            loss = self.loss_f(prediction, y) 
            loss.backward() 
            self.opt.step()
            total_loss += loss.item() * X.size(0) 
        self.avg_train_loss = total_loss/len(self.train_loader.dataset)
        
    def _validation(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                prediction = self.model(X)
                loss = self.loss_f(prediction, y)
                total_loss += loss.item() * X.size(0)
        self.avg_val_loss = total_loss/len(self.val_loader.dataset)



    def train(self, patience=10):
        counter = 0
        for epoch in range(self.epochs):
            self._training()
            self._validation()

            #print(f'Validation loss: {self.avg_val_loss}. Train loss: {self.avg_train_loss}.')

            if self.avg_val_loss < self.best_val_loss:
                counter = 0
                self.best_val_loss = self.avg_val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                counter +=1
                if counter == patience:
                    break
        self.model.load_state_dict(torch.load("best_model.pth", weights_only=True))



    def tester(self, boundary, metrics=['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']):
        metrics_functions = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1_score,
            'roc_auc': roc_auc_score 
        }

        self.model.eval()
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                prediction = torch.sigmoid(self.model(X)) 
                y_true_list.append(y.cpu())
                y_pred_list.append(prediction.cpu())

        y_true = torch.cat(y_true_list, dim=0).numpy()
        y_pred = torch.cat(y_pred_list, dim=0).numpy()
        y_class = (y_pred >= boundary).astype(int)

        results = {}
        for metric in metrics:
            if metric in metrics_functions:
                func = metrics_functions[metric]
                score = func(y_true, y_pred if metric == 'roc_auc' else y_class)
                results[metric] = score
                print(f"{metric} : {score}")

        print("\n--- Macierz Pomyłek ---")
        print(confusion_matrix(y_true, y_class))

        return results