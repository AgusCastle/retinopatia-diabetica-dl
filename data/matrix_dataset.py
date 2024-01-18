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

        # if self.set == 'test': # Esto fue porque se duplicaron dos predicciones jajaja por que no se 
        #     #pero asi fue rapido arrglarlo ya despues genero el json chido jaja
        #     aux = self.data[index]['matrix']
        #     aux.pop(3)
        #     aux.pop(1)
        #     return torch.permute(torch.FloatTensor(aux), (1, 0)), self.data[index]['label'], self.data[index]['filename']
        aux = self.data[index]['matrix']
        #aux =[aux[1], aux[2], aux[4], aux[6], aux[7]]
        return torch.permute(torch.FloatTensor(aux), (1, 0)), self.data[index]['label'], self.data[index]['filename']
    def __len__(self):
        return self.data.__len__()
    
class MatrixDatasetCustom(Dataset):
    def __init__(self, root, set, selects : list):
        super()
        self.root = root
        self.selects = selects

        with open(root, 'r') as file:
            data = json.load(file)

        self.set = set
        self.data = data

    def __getitem__(self, index):

        aux = []

        for i, vectores in enumerate(self.data[index]['matrix']):
            if i  in self.selects:
                aux.append(vectores)
                
        return torch.permute(torch.FloatTensor(aux), (1, 0)), self.data[index]['label'], self.data[index]['filename']
    
    def __len__(self):
        return self.data.__len__()
    

class MatrixDatasetCustomUnion(Dataset):
    def __init__(self, roots: [], set: str, selects : list):
        super()
        self.roots = roots
        self.selects = selects

        data = []

        for root in self.roots:
            with open(root, 'r') as file:
                if len(data) == 0:
                    data = json.load(file)
                else:
                    data = agregarData(json.load(file))

        self.set = set
        self.data = data

    def __getitem__(self, index):

        aux = []

        for i, vectores in enumerate(self.data[index]['matrix']):
            if i  in self.selects:
                aux.append(vectores)
                
        return torch.permute(torch.FloatTensor(aux), (1, 0)), self.data[index]['label'], self.data[index]['filename']
    
    def __len__(self):
        return self.data.__len__()
    
def agregarData(old : [], new: [] ):

    for obj in old:
        nombre = obj['filename']
        obj['matrix'].extends(getContainsDict(new, nombre))
    
    return old

def getContainsDict(array, filename):
    for i, obj in enumerate(array):
        if obj['filename'] == filename:
            return obj['matrix']
    return -1