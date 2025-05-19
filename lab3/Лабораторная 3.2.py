import pandas as pd
import torch



df = pd.read_csv('data.csv')
print(df.head())


features = df.iloc[:, [0, 2, 3]].values
labels = df.iloc[:, 4].values


X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor([1 if label == "Iris-setosa" else -1 for label in labels], dtype=torch.float32)


w = torch.rand(4, dtype=torch.float32, requires_grad=True)

speed_obuch = 0.01
num_epochs = 100

def neuron(w, x):
    return 1 if (w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[0]) >= 0 else -1

for epoch in range(num_epochs):
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        prediction = neuron(w, xi)
        with torch.no_grad():
            w_new = w.clone()
            w_new[1:] = w[1:] + speed_obuch * (target - prediction) * xi
            w_new[0] = w[0] + speed_obuch * (target - prediction)
            w.copy_(w_new)


correct = 0
for i in range(len(X)):
    if neuron(w, X[i]) == y[i]:
        correct += 1


