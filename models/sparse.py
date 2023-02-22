import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json

class SparseFusion(nn.Module):
    def __init__(self, n_classes) -> None:
        super(SparseFusion, self).__init__()
        W = torch.randn(9, 5, dtype=torch.float32)
        W = W.to(torch.device(0))
        self.W = nn.Parameter(W)

        self.i = torch.eye(n_classes)
        self.i = self.i.to(torch.device(0))

        W2 = torch.randn(5, dtype=torch.float32)
        W2 = W2.to(torch.device(0))
        self.W2 = nn.Parameter(W2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.matmul(x, self.W)
        x = torch.mul(x, self.i)
        x = torch.diagonal(x, dim1= -1, dim2= -2)
        x = torch.mul(x, self.W2)
        return self.softmax(x)

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
            aux = self.data[index]['matrix']
            aux.pop(3)
            aux.pop(1)

            # subset = [aux[1], aux[6], aux[2], aux[5], aux[0]]

            return torch.permute(torch.FloatTensor(aux), (1, 0)), self.data[index]['label'], self.data[index]['filename']
            # return torch.FloatTensor(subset) , self.data[index]['label'], self.data[index]['filename']

        return torch.permute(torch.FloatTensor(self.data[index]['matrix']), (1, 0)), self.data[index]['label'], self.data[index]['filename']
        # aux = self.data[index]['matrix']

        #subset = [aux[1], aux[6], aux[2], aux[5], aux[0]]

        # return torch.FloatTensor(subset) , self.data[index]['label'], self.data[index]['filename']

    def __len__(self):
        return self.data.__len__()

def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale

def save_checkpoint(epoch, model, optimizer, filename, model_str = 'model'):
        state = {
            'str': model_str,
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer
        }

        torch.save(state, filename)

def eval(model, set = 'valid', flag = False):
    
    model = model.eval()
    model = model.to(torch.device(0))
    dataset = MatrixDataset('/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/JSONFiles/DDR_M/DDR_{}.json'.format(set), set)
    dataloader = DataLoader(dataset, shuffle= True, batch_size=256)
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    pred = []
    gt = []
    print('Evaluando set: {}'.format(set))
    for _, batch in process_bar:

        matrix, labels, _ = batch
        gt.extend(int(label) for label in labels)

        labels = labels.squeeze()
        matrix = matrix.to(torch.device(0))
        labels = labels.to(torch.device(0))

        x = model(matrix)
        pred.extend(int(torch.argmax(tensor)) for tensor in x)

    cfm = confusion_matrix(gt,pred).tolist()

    if flag: 
        if set == 'valid':
            img_total = [1253, 126, 895, 47, 182]
        if set == 'test':
            img_total = [1880, 189, 1344, 71, 275]
        print('acc: {}'.format(accuracy_score(gt, pred)))
        for i in range(5):
            res = float(cfm[i][i])/ img_total[i]
            
            print('{}: {}'.format(i, res))
        return

    return accuracy_score(gt, pred)
    

    
if __name__ == '__main__':
    model = SparseFusion(5)
    model = model.to(torch.device(0))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model.train()
    loss_total = 0.
    dataset = MatrixDataset('/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/JSONFiles/DDR_M/DDR_train.json', 'train')
    dataloader = DataLoader(dataset, shuffle= True, batch_size=512)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', patience=100, factor=0.1, verbose=True)
    for epoch in range(700):
        process_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, batch in process_bar:

            matrix, label, _ = batch
            label = label.squeeze()
            matrix = matrix.to(torch.device(0))
            label = label.to(torch.device(0))

            x = model(matrix)

            loss = criterion(x, label)
            loss_total += float(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            process_bar.set_description_str(
            'Epoch {} : Loss: {:.3f}'.format(epoch + 1, float(loss)), True)
        
        acc = eval(model)

        print('Acc obtenido : {:.3f}'.format(acc * 100))
        scheduler.step(acc)
    
    eval(model, 'test', True)
    eval(model, 'valid', True)
    
    save_checkpoint(2000, model, optimizer, '/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/models/sparse_models/{}_{}.pth'.format('sparse', 'arr_5'))