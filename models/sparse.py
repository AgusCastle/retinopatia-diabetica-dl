import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
import argparse
import sys
from utils.metrics import MericsEvaluation
from utils.save_info import Util
from utils.services.google_service import GoogleService
from data.matrix_dataset import MatrixDatasetCustom, MatrixDataset, MatrixDatasetCustomUnion
from utils.cost_lost_sensitive import CostSensitiveRegularizedLoss

class SparseFusion(nn.Module):
    def __init__(self, n_classes, device, n = 9) -> None:
        super(SparseFusion, self).__init__()
        W = torch.randn(n, 5, dtype=torch.float32)
        W = W.to(device)
        self.W = nn.Parameter(W)

        self.i = torch.eye(n_classes)
        self.i = self.i.to(device)

        W2 = torch.randn(5, dtype=torch.float32)
        W2 = W2.to(device)
        self.W2 = nn.Parameter(W2)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.matmul(x, self.W)
        x = torch.mul(x, self.i)
        x = torch.diagonal(x, dim1= -1, dim2= -2)
        x = torch.mul(x, self.W2)
        return self.softmax(x)

def eval(dataloader, model, device, set = 'test', flag = False, messidor=False, savename='', selected = []):
    
    model = model.eval()
    model = model.to(device)
    if messidor or True:
        dataset = MatrixDatasetCustom('{}{}.json'.format(dataloader,set), set, selected)
    else:
        dataset = MatrixDataset('/home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/JSONFiles/DDR_M/DDR_{}.json'.format(set), set)
    dataloader = DataLoader(dataset, shuffle= True, batch_size=128)
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    pred = []
    gt = []
    print('Evaluando set: {}'.format(set))
    for _, batch in process_bar:

        matrix, labels, _ = batch
        gt.extend(int(label) for label in labels)

        labels = labels.squeeze()
        matrix = matrix.to(torch.device(1))
        labels = labels.to(torch.device(1))

        x = model(matrix)
        pred.extend(int(torch.argmax(tensor)) for tensor in x)

    metrics = MericsEvaluation(pred, gt, 'test')

    if flag:
        gs = GoogleService()

        if messidor:
            info_r = [savename, '-', 'ddr', '-', set]
        else:
            info_r = [savename, '-', 'ddr', '-', set]
        info_r.extend(metrics.getall())
        gs.insertRowToSheet(info_r)

    return metrics.accuracy()

def evalUnion(dataloader, model, device, set = 'test', flag = False, messidor=False, savename='', selected = []):
    
    model = model.eval()
    model = model.to(device)

    rt_eval = []
    for i in range(3):
        rt_eval.append('{}{}.json'.format(dataloader[i],set))
    
    dataset = MatrixDatasetCustomUnion(rt_eval, set, selected)
    dataloader = DataLoader(dataset, shuffle= True, batch_size=128)
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    pred = []
    gt = []
    print('Evaluando set: {}'.format(set))
    for _, batch in process_bar:

        matrix, labels, _ = batch
        gt.extend(int(label) for label in labels)

        labels = labels.squeeze()
        matrix = matrix.to(torch.device(1))
        labels = labels.to(torch.device(1))

        x = model(matrix)
        pred.extend(int(torch.argmax(tensor)) for tensor in x)

    metrics = MericsEvaluation(pred, gt, 'test')

    if flag:
        gs = GoogleService()

        if messidor:
            info_r = [savename, '-', 'ddr', '-', set]
        else:
            info_r = [savename, '-', 'ddr', '-', set]
        info_r.extend(metrics.getall())
        gs.insertRowToSheet(info_r)

    return metrics.accuracy()
    
def evalSnf(dataloader ,load_model, device):
    
    device = torch.device(device)
    model = torch.load(load_model, map_location=device)['model']

    #eval(model, device, 'valid', True, False, load_model)
    eval(dataloader,model, device, 'test', True, False, load_model)



def trainEval(dataloader_str ,lr=0.9, factor_lr=0.1, patience=100, epochs= 700, batch_size=512, device=1, save_name='SNF', evals=False, path_model='', loss_sensitive =  False ,loss_mode = 0):

    device = torch.device(device)

    if not evals:
        model = SparseFusion(5, device)
    else:
        checkp = torch.load(path_model, map_location=device)
        model = checkp['model']

    model = model.to(device)

    if evals:
        eval(model, device, 'test', True)
        eval(model, device, 'valid', True)
        sys.exit()
    
    best_acc = 0

    if loss_sensitive:
        criterion = CostSensitiveRegularizedLoss(5, reduction='mean', mode=loss_mode, base_loss='ce')
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    loss_total = 0.
    dataset = MatrixDataset('{}train.json'.format(dataloader_str), 'train')
    dataloader = DataLoader(dataset, shuffle= True, batch_size=batch_size)
    scheduler = ReduceLROnPlateau(
        optimizer, 'max', patience=patience, factor=factor_lr, verbose=True, min_lr=1e-5)
    for epoch in range(epochs):
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
        
        # hola guapo, te extaño. No veas porno solito! JAJAJAJA no , como les fue con el giby?
        acc = eval(dataloader_str,model,device,'valid')

        print('Acc obtenido : {:.3f}'.format(acc * 100))
        scheduler.step(round(acc, 4))

        if best_acc < acc:
            best_acc = acc
            Util.save_checkpoint(epochs, model, optimizer, save_name + 'best.pth', 'SNF', {'lr': lr, 'patience': patience, 'factor': factor_lr, 'epochs': epochs})
    
    eval(dataloader_str,model, device, 'valid', True, savename=save_name)
    eval(dataloader_str,model, device, 'test', True,savename= save_name)
    #eval(model, device, 'test', True, True, savename= save_name)
    eval(dataloader_str, torch.load(save_name + 'best.pth')['model'], device, 'valid', True, False, savename= save_name)
    
    
    Util.save_checkpoint(epochs, model, optimizer, save_name, 'SNF', {'lr': lr, 'patience': patience, 'factor': factor_lr, 'epochs': epochs})


def trainEvalCustom(dataloader_str ,lr=0.9, factor_lr=0.1, patience=100, epochs= 700, batch_size=512, device=1, save_name='SNF', evals=False, path_model='', loss_sensitive =  False ,loss_mode = 0, selected = []):

    device = torch.device(device)


    if not evals:
        model = SparseFusion(5, device, n=len(selected))
    else:
        checkp = torch.load(path_model, map_location=device)
        model = checkp['model']

    model = model.to(device)

    if evals:
        eval(model, device, 'test', True, selected=selected)
        eval(model, device, 'valid', True, selected=selected)
        sys.exit()
    
    best_acc = 0

    if loss_sensitive:
        criterion = CostSensitiveRegularizedLoss(5, reduction='mean', mode=loss_mode, base_loss='ce')
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    loss_total = 0.
    dataset = MatrixDatasetCustom('{}train.json'.format(dataloader_str), 'train', selects=selected)
    dataloader = DataLoader(dataset, shuffle= True, batch_size=batch_size)
    scheduler = ReduceLROnPlateau(
        optimizer, 'max', patience=patience, factor=factor_lr, verbose=True, min_lr=1e-5)
    for epoch in range(epochs):
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
        
        acc = eval(dataloader_str,model,device,'valid', selected= selected)

        print('Acc obtenido : {:.3f}'.format(acc * 100))
        scheduler.step(round(acc, 4))

        if best_acc < acc:
            best_acc = acc
            Util.save_checkpoint(epochs, model, optimizer, save_name + 'best.pth', 'SNF', {'lr': lr, 'patience': patience, 'factor': factor_lr, 'epochs': epochs})
    
    eval(dataloader_str,model, device, 'valid', True, savename=save_name,selected= selected)
    eval(dataloader_str,model, device, 'test', True,savename= save_name,selected= selected)
    #eval(model, device, 'test', True, True, savename= save_name)
    eval(dataloader_str, torch.load(save_name + 'best.pth')['model'], device, 'valid', True, False, savename= save_name, selected=selected)
    eval(dataloader_str, torch.load(save_name + 'best.pth')['model'], device, 'test', True, False, savename= save_name, selected=selected)
    
    
    Util.save_checkpoint(epochs, model, optimizer, save_name, 'SNF', {'lr': lr, 'patience': patience, 'factor': factor_lr, 'epochs': epochs})

def trainEvalCustomMultiples(dataloader_str, dataloader_str_2, dataloader_str_3 ,lr=0.9, factor_lr=0.1, patience=100, epochs= 700, batch_size=512, device=1, save_name='SNF', evals=False, path_model='', loss_sensitive =  False ,loss_mode = 0, selected = []):

    device = torch.device(device)



    if not evals:
        model = SparseFusion(5, device, n=len(selected))
    else:
        checkp = torch.load(path_model, map_location=device)
        model = checkp['model']

    model = model.to(device)

    if evals:
        eval(model, device, 'test', True, selected=selected)
        eval(model, device, 'valid', True, selected=selected)
        sys.exit()
    
    best_acc = 0

    if loss_sensitive:
        criterion = CostSensitiveRegularizedLoss(5, reduction='mean', mode=loss_mode, base_loss='ce')
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    loss_total = 0.

    roots = [dataloader_str, dataloader_str_2, dataloader_str_3]

    rt_train = []
    for i in range(3):
        rt_train.append('{}train.json'.format(roots[i]))

    dataset = MatrixDatasetCustomUnion(rt_train, 'train', selects=selected)
    dataloader = DataLoader(dataset, shuffle= True, batch_size=batch_size)
    scheduler = ReduceLROnPlateau(
        optimizer, 'max', patience=patience, factor=factor_lr, verbose=True, min_lr=1e-5)
    for epoch in range(epochs):
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
        
        acc = evalUnion(roots,model,device,'valid', selected= selected)

        print('Acc obtenido : {:.3f}'.format(acc * 100))
        scheduler.step(round(acc, 4))

        if best_acc < acc:
            best_acc = acc
            Util.save_checkpoint(epochs, model, optimizer, save_name + 'best.pth', 'SNF', {'lr': lr, 'patience': patience, 'factor': factor_lr, 'epochs': epochs})
    
    evalUnion(roots,model, device, 'valid', True, savename=save_name,selected= selected)
    evalUnion(roots,model, device, 'test', True,savename= save_name,selected= selected)
    #eval(model, device, 'test', True, True, savename= save_name)
    evalUnion(roots, torch.load(save_name + 'best.pth')['model'], device, 'valid', True, False, savename= save_name, selected=selected)
    evalUnion(roots, torch.load(save_name + 'best.pth')['model'], device, 'test', True, False, savename= save_name, selected=selected)
    
    
    Util.save_checkpoint(epochs, model, optimizer, save_name, 'SNF', {'lr': lr, 'patience': patience, 'factor': factor_lr, 'epochs': epochs})