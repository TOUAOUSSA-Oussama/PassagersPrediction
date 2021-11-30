## L'objectif est de predire le nombre des passageres dans un aeroport international
## le nombre des passagers est donner en unite de 1000
## la base de donnee disponible est d'un intervale de janvier 1949 jusqu'a decembre 1960
## avec 144 observations

## 1 - import necessary labraries :
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import MinMaxScaler

## 2 - Load and preprocess the Data :
# load data :
dataset = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python') # remarque : au lieu d'apprendre t = mois, annee, on peut dire que t=0 correspond au janvier 1949, t=1 au feverier 1949..... 
# la convetir en numpy array et de type float32
dataset = np.array(dataset).astype(np.float32)
# Noramalisation des donnees : parce que le nombre des passagers du premier mois est trop faible par rapport aux mois recents
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset.reshape(-1, 1))
# affichage de la taille des donnees : 
print("taille = ", len(dataset)) # 144 observations ou chaque observation = nbre de passengers pendant un mois t 
# affichage de la courbe des donnees
plt.plot(dataset)
plt.show()

## 3 - Definition du modele de classification :
class LSTMModel(nn.Module):
    def __init__(self, n_hidden=80):
        super(LSTMModel, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)
    
    def forward(self, input, future=0):
        n_samples = 1

        outputs = []
        
        h_t1 = torch.zeros(1, self.n_hidden, dtype=torch.float32)
        c_t1 = torch.zeros(1, self.n_hidden, dtype=torch.float32)

        h_t2 = torch.zeros(1, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(1, self.n_hidden, dtype=torch.float32)

        # forward sur chaque observations de l'entrainement
        for input_t in input.split(1, dim=0):
            h_t1, c_t1 = self.lstm1(input_t , (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        # pour le cas de prediction
        for i in range(future):
            h_t1, c_t1 = self.lstm1(output , (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        # convertir outputs en un tenseur
        outputs = torch.cat(outputs, dim=1)

        return outputs

if __name__ == "__main__":
    ## 4 - split into train and test data : 80 % pour l'entrainement et 20 % pour la prediction
    train_size = int(len(dataset)*0.8)
    # ensemble d'entrainement :
    train_x = dataset[:train_size-1] # correspond a l'intant t-1
    train_y = dataset[1:train_size] # correspond a l'instant t
    # ensemble de test :
    test_x = dataset[train_size:len(dataset)-1]
    test_y = dataset[train_size+1:len(dataset)]
    # Conversion en des tenseurs :
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    ## 5 - Defintion du modele :
    modele = LSTMModel()

    ## 6 - Defintion du loss et d'optimiseur :
    # Hyperparamters :
    learning_rate  = 0.01;
    nbr_epochs = 200;

    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(modele.parameters(), lr=learning_rate)

    ## 7 - Entrainement du modele :
    losses = []
    for i in range(nbr_epochs):
        
        def closure():
            optimizer.zero_grad() # on vide le gradient de l'optimiseur
            out = modele(train_x)
            loss = criterion(out, train_y)
            loss.backward() # backward pour calculer le graduit
            losses.append(loss.item())
            if (i%20) == 0:
                print("Epoch : ", i+1)
                print("Loss = ", loss.item())

            return loss
        optimizer.step(closure)

        
        # Affichage de la courbe d'entrainement :
    plt.title("Loss of training")
    plt.plot(losses)
    plt.show()
    ## 8 - Evaluation du modele :
    with torch.no_grad():
        pred = modele(test_x)
        loss = criterion(pred, test_y)
        print("Test Loss : ", loss.item())

## 9 - Prediction des 24 mois qui suivent :
with torch.no_grad():
    y = torch.from_numpy(dataset)
    y_test = modele(y, future=24)
pred = y_test.detach().numpy()
pred = pred.reshape(-1, 1)
print(y_test.shape)
plt.plot(pred, color='r')
plt.show()