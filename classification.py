import os
from tqdm import tqdm

from torch import nn
import torch

from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

#Hyperparameters

optimizer = torch.optim.Adam
learning_rate = 10e-4
loss_fn = torch.nn.MSELoss()
n_epochs = 10
save_model_file_path= 'saved_model/saved_model'

class Classifier(nn.Module):
    '''
    Base class for the two classifiers. Implements train loop and performance evaluation methods, together with other tool functions.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.main_func : nn.Module = None


    def forward(self, x) -> torch.Tensor:
        pass


    def train(self,
            TrainLoader : DataLoader,):
    
        #Instanciates optimizer on model parameters
        
        optimizer_inst = optimizer(self.parameters(), lr = learning_rate)

        for epoch in range(1, n_epochs+1):
            print(10*'-'+f'[ Epoch {epoch} ]'+10*'-')
            loop = tqdm(TrainLoader)
            super().train(mode = True)

            for batch_idx, (data, labels) in enumerate(loop):
                labels = one_hot(labels, num_classes = 10).float()
                predictions = self.forward(data)
                loss: torch.Tensor = loss_fn(labels, predictions)
            
                optimizer_inst.zero_grad()
                loss.backward()
                optimizer_inst.step()

                loop.set_postfix(loss = loss.item())
            #Ecvery ten epochs, saves current state of model
            if epoch % 10 == 0:
                torch.save(obj =
                            {
                            'model_state' : self.main_func.state_dict()
                            },
                            f = save_model_file_path)


    def compute_acc(self,
                Loader : DataLoader,
                ):
        '''
        Compute accuracy of model over dataset covered by passed DataLoader
        '''
        super().train(mode = False)
        count = 0
        pos = 0
        Loader.batch_size = 1
        for data, label in Loader:
            prediction : torch.Tensor = self.forward(data)
            predicted_label = self.to_label(prediction)
            if predicted_label[0] == label.item():
                pos+= 1
            count+=1
        return pos/count
    
    
    @staticmethod
    def to_label(one_hot : torch.Tensor):
        '''
        Takes a one-hot encoding and returns the corresponding label
        '''
        n = one_hot.shape[0]
        predicted_label = [one_hot[i,:].argmax().item() for i in range(n)]
        return predicted_label
    



class ClassifierCNN(Classifier):

    '''
    CNN classifier
    '''
    def __init__(self):
        super().__init__()
        self.main_func = nn.Sequential(
            nn.Conv2d(in_channels = 1,
                      out_channels = 32,
                      kernel_size = 5,
                      stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 32,
                      out_channels = 64,
                      kernel_size = 5,
                      stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Flatten(),

            nn.Linear(in_features = 1024,
                      out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128,
                      out_features = 10),
            nn.Softmax(dim = 1)

        )
        

    def forward(self, x):
        return self.main_func(x)

    
class ClassifierLinear(Classifier):

    '''
    Linear stack classifier
    '''
    def __init__(self):
        super().__init__()
        self.main_func = nn.Sequential(
            
            nn.Flatten(),
            nn.Linear(in_features = 28*28, out_features = 28*28),
            nn.ReLU(),
            nn.Linear(in_features = 28*28, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 10),
            nn.Softmax(dim = 0)

        )
        

    def forward(self, x):
        return self.main_func(x)