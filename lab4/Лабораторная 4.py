#я шестая по списку, решала задачу на предсказание дохода

import pandas as pd
import torch
import torch.nn as nn
df = pd.read_csv('dataset_simple.csv')
X = torch.tensor(df[['age']].values, dtype=torch.float32)
y = torch.tensor(df[['income']].values, dtype=torch.float32)
class NNetRegression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, X):
        return self.layers(X)

inputSize = X.shape[1] 
hiddenSize = 30
outputSize = 1
net = NNetRegression(inputSize, hiddenSize, outputSize)
lossFn = nn.L1Loss()  
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
epohs = 1000
for i in range(epohs):
    pred = net.forward(X)  
    loss = lossFn(pred.squeeze(), y)  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  
    
    if i % 10 == 0:
        print(f'Ошибка на {i+1} итерации: {loss.item():.6f}')
with torch.no_grad():
    pred = net.forward(X)

print('\nПредсказания:')
print(pred[0:10])
err = torch.mean(abs(y - pred.T).squeeze())  
print('\nОшибка (MAE):')
print(err)

with torch.no_grad():
    y1 = net.forward(torch.Tensor([40]))
    y2 = net.forward(torch.Tensor([50]))
print(y1,y2)