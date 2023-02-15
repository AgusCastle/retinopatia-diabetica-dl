from torch.utils.data import Dataset
import json
import torch

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