import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

class SparseFusion(nn.Module):
    def __init__(self, n_classes) -> None:
        super(SparseFusion, self).__init__()
        W = torch.randn(9, 5, dtype=torch.float32, requires_grad=True)
        W = W.to(torch.device(0))
        self.W = nn.Parameter(W)
        self.i = torch.eye(n_classes)
        self.i = self.i.to(torch.device(0))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = torch.matmul(x, self.W)
        x = torch.matmul(x, self.i)
        x = torch.diagonal(x, 0) # Se que esta mal pero lo intente
        return x

class MatrixDataset(Dataset):
    def __init__(self, root, set):
        super()
        self.root = root

        with open(root, 'r') as file:
            data = json.load(file)

        self.set = set
        self.data = data

    def __getitem__(self, index):

        if self.set == 'test': # Esto fue porque se duplicaron dos predicciones jajaja por que no se 
            #pero asi fue rapido arrglarlo ya despues genero el json chido jaja
            aux = self.data['matrix']
            aux.pop(3)
            aux.pop(1)
            return torch.FloatTensor(aux), self.data[index]['label'], self.data[index]['filename']

        return torch.FloatTensor(self.data[index]['matrix']), self.data[index]['label'], self.data[index]['filename']

    def __len__(self):
        return self.data.__len__()
    
if __name__ == '__main__':
    model = SparseFusion(5)
    model = model.to(torch.device(0))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model.train()

    dataset = MatrixDataset('/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/JSONFiles/DDR_M/DDR_valid.json', 'valid')
    dataloader = DataLoader(dataset, shuffle= True, batch_size=256)
    for epoch in range(100):
        process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for _, batch in process_bar:

            matrix, label, _ = batch
            label = label.squeeze()
            matrix = matrix.to(torch.device(0))
            label = label.to(torch.device(0))

            x = model(matrix)

            print(x.size())
            print(label.size())

            loss = criterion(x, label)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()