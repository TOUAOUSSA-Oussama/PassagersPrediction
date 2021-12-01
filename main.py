## 1 - import necessary labraries :
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

## 2 - Load and preprocess the Data :
# load data :
dataset = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python') 
# la convetir en numpy array et de type float32
dataset = np.array(dataset).astype(np.float32)
# Noramalisation des donnees : 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.reshape(-1, 1))
# affichage de la taille des donnees : 
print("taille = ", dataset.shape) # 144 observations ou chaque observation = nbre de passengers pendant un mois t 
# affichage de la courbe des donnees
plt.plot(dataset)
plt.show()


## 3 - Preparation des donnees :
def sliding_Window(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length):
        x_seq = data[i:i+seq_length] # prendre les 4 donnees pour predire celle de la 5 ieme
        y_seq = data[i+seq_length]
        x.append(x_seq)
        y.append(y_seq)
    return x, y

seq_length = 4
x, y = sliding_Window(dataset, seq_length)
x = np.array(x)
y = np.array(y)
print("Data preparation :")
print("X size : ", x.shape) # (140, 4, 1)
print("Y size : ", y.shape) # (140, 1)


## 3 - Definition du modele de classification :
class LSTMModel(nn.Module):
    def  __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # print(x.shape) = 114, 4, 1
        _, (out, _) = self.lstm(x, (h_0, c_0))
        out = out.view(-1, self.hidden_size)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    ## 4 - split into train and test data : 80 % pour l'entrainement et 20 % pour la prediction
    train_size = int(len(dataset)*0.8)
    # ensemble d'entrainement :
    train_x = x[:train_size-1] # correspond a l'intant t-1
    train_y = y[1:train_size] # correspond a l'instant t
    # ensemble de test :
    test_x = x[train_size:len(dataset)-1]
    test_y = y[train_size:len(dataset)]
    # Conversion en des tenseurs :
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    # Hyperparamters :
    learning_rate  = 0.02;
    nbr_epochs = 6000;
    input_size = 1
    hidden_size = 2
    num_layers = 1
    num_classes = 1

    ## 5 - Defintion du modele :
    modele = LSTMModel(num_classes, input_size, hidden_size, num_layers)

    ## 6 - Defintion du loss et d'optimiseur :
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modele.parameters(), lr=learning_rate)

    ## 7 - Entrainement du modele :
    losses = []
    for i in range(nbr_epochs):
        
        optimizer.zero_grad() # on vide le gradient de l'optimiseur
        # print(train_x.shape) = torch.Size([114, 4, 1])
        out = modele(train_x)
        loss = criterion(out, train_y)
        loss.backward()
        losses.append(loss.item())
        
        optimizer.step()

        if (i%100) == 0:
            print("Epoch : ", i+1)
            print("Loss = ", loss.item())

        
    # Affichage de la courbe d'entrainement :
    plt.title("Loss of training")
    plt.plot(losses)
    plt.show()

    ## 8 - Evaluation du modele :
    with torch.no_grad():
        out = modele(test_x)
        loss = criterion(out, test_y)
        print("Test Loss = ", loss.item())

## Affichage de la prediction sur la data toute entiere :
x = torch.from_numpy(x)
all_predict = modele(x)
all_predict = all_predict.data.numpy()
# Enlever la normalisation
all_predict = scaler.inverse_transform(all_predict)
y = scaler.inverse_transform(y)
# Tracer la ligne qui divise train set et test set:
plt.axvline(x=train_size, c='r', linestyle='--')
# Tracer les courbe x et y:
plt.plot(y)
plt.plot(all_predict)
plt.suptitle('Prediction :')
plt.show()
