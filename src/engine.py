import torch
import torch.nn as nn 



class Engine():
    def __init__(self, optimizer, model, device):
        self.optimizer = optimizer
        self.model = model
        self.device = device 
        
        
    def loss_fn(outputs, targets):
        return nn.BCELoss()(outputs, targets.view(-1, 1))
    
    
    def train_fn(self, train_loader):
        target_values, output_values = [], []
        
        features = 
        targets = 

        self.model.train()



        target_values.append()
        output_values.append()        
        return target_values, output_values
        
        
        self.model(features)
        self.optimizer.step()


    def eval_fn(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            