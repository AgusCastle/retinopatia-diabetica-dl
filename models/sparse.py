import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
import argparse
import sys

class SparseFusion(nn.Module):
    def __init__(self, n_classes, device) -> None:
        super(SparseFusion, self).__init__()
        W = torch.randn(9, 5, dtype=torch.float32)
        W = W.to(device)
        self.W = nn.Parameter(W)

        self.i = torch.eye(n_classes)
        self.i = self.i.to(device)

        W2 = torch.randn(5, dtype=torch.float32)
        W2 = W2.to(device)
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

        if self.set == 'test':
            aux = self.data[index]['matrix']
            aux.pop(3)
            aux.pop(1)

            return torch.permute(torch.FloatTensor(aux), (1, 0)), self.data[index]['label'], self.data[index]['filename']
            
        return torch.permute(torch.FloatTensor(self.data[index]['matrix']), (1, 0)), self.data[index]['label'], self.data[index]['filename']
 
    def __len__(self):
        return self.data.__len__()

def save_checkpoint(epoch, model, optimizer, filename, model_str = 'model'):
        state = {
            'str': model_str,
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer
        }

        torch.save(state, filename)

def eval(model, device, set = 'valid', flag = False):
    
    model = model.eval()
    model = model.to(device)
    dataset = MatrixDataset('/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/JSONFiles/DDR_M/DDR_{}.json'.format(set), set)
    dataloader = DataLoader(dataset, shuffle= True, batch_size=512)
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

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--lr', default= 0.1, type=float)
    parser.add_argument('--factor_lr', default= 0.1, type=float)
    parser.add_argument('--patience', default= 100, type=int)
    parser.add_argument('--epochs', default= 700, type=int)
    parser.add_argument('--batch_size', default= 512, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--save_name')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--path_model', type=str)

    args = parser.parse_args()

    device = torch.device(args.device)

    if not args.eval:
        model = SparseFusion(5, device)
    else:
        checkp = torch.load(args.path_model, map_location=device)
        model = checkp['model']
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(count_parameters(model))

    model = model.to(device)

    if args.eval:
        eval(model, device, 'test', True)
        eval(model, device, 'valid', True)
        sys.exit()
        

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    model.train()
    loss_total = 0.
    dataset = MatrixDataset('/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/JSONFiles/DDR_M/DDR_train.json', 'train')
    dataloader = DataLoader(dataset, shuffle= True, batch_size=args.batch_size)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', patience=args.patience, factor=args.factor_lr, verbose=True, min_lr=1e-4)
    for epoch in range(args.epochs):
        process_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, batch in process_bar:

            matrix, label, _ = batch
            label = label.squeeze()
            matrix = matrix.to(device)
            label = label.to(device)

            x = model(matrix)

            loss = criterion(x, label)
            loss_total += float(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            process_bar.set_description_str(
            'Epoch {} : Loss: {:.3f}'.format(epoch + 1, float(loss)), True)
        
        acc = eval(model,device)

        print('Acc obtenido : {:.3f}'.format(acc * 100))
        scheduler.step(round(acc, 4))
    
    eval(model, device, 'valid', True)
    eval(model, device, 'test', True)
    
    
    save_checkpoint(2000, model, optimizer, '/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/models/sparse_models/{}.pth'.format(args.save_name))