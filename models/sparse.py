import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from tqdm import tqdm
import json

class SparseFusion(nn.Module):
    def __init__(self, n_classes) -> None:
        super(SparseFusion, self).__init__()
        W = torch.zeros(9, 5, dtype=torch.float32, requires_grad=True)
        W = W.to(torch.device(0))
        self.W = nn.Parameter(W)
        self.i = torch.eye(n_classes)
        self.i = self.i.to(torch.device(0))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.matmul(x, self.W)
        x = torch.matmul(x, self.i)
        x = torch.diagonal(x, dim1= -1, dim2= -2)
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
            return torch.permute(torch.FloatTensor(aux), (1, 0)), self.data[index]['label'], self.data[index]['filename']

        return torch.permute(torch.FloatTensor(self.data[index]['matrix']), (1, 0)), self.data[index]['label'], self.data[index]['filename']

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
    
if __name__ == '__main__':
    model = SparseFusion(5)
    model = model.to(torch.device(0))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    loss_total = 0.
    dataset = MatrixDataset('/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/JSONFiles/DDR_M/DDR_train.json', 'train')
    dataloader = DataLoader(dataset, shuffle= True, batch_size=256)
    for epoch in range(10000):
        process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        if epoch % 5000 == 0 and epoch != 0:
            print('decay')
            adjust_learning_rate(optimizer, 0.1)

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
    
    save_checkpoint(10000, model, optimizer, '/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/models/sparse_models/{}_{}.pth'.format('sparse', 5))
    #model = torch.load('/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/models/sparse_models/sparse_3.pth')
    #model =model['model']
    model = model.eval()
    model = model.to(torch.device(0))
    dataset = MatrixDataset('/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/JSONFiles/DDR_M/DDR_test.json', 'test')
    dataloader = DataLoader(dataset, shuffle= True, batch_size=256)
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    pred = []
    gt = []
    for _, batch in process_bar:

        matrix, labels, _ = batch
        gt.extend(int(label) for label in labels)

        labels = labels.squeeze()
        matrix = matrix.to(torch.device(0))
        labels = labels.to(torch.device(0))

        x = model(matrix)
        pred.extend(int(torch.argmax(tensor)) for tensor in x)

    cfm = confusion_matrix(gt, pred).tolist()
    print('Acc: {}'.format(accuracy_score(gt, pred)))
    
    set = 'test'
    if set == 'valid':
        img_total = [1253, 126, 895, 47, 182]
    if set == 'test':
        img_total = [1880, 189, 1344, 71, 275]
    
    for i in range(5):
        res = float(cfm[i][i])/ img_total[i]
        print('{}: {}'.format(i, res))