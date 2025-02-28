import torch
from torch.utils.data import DataLoader
from data.drdataset import DrDataset
from tqdm import tqdm
from utils.save_info import Util
from pathlib import Path
from utils.metrics import MericsEvaluation
from utils.services.google_service import GoogleService
import re

def evalModelOneDataset(model_load: str, dataloader: str = 'JSONFiles/messidor2/messidor2_', set : str = 'test',devicef = 1, message= None):

    device = torch.device(devicef)
    checkpoint = torch.load(model_load, map_location=device)

    epoch, model = checkpoint['epoch'], checkpoint['model']
    model.to(device)
    print('Especificaciones del Modelo')
    print(checkpoint['init'])
    print('Ultima epoca: {} -> {}'.format(epoch, checkpoint['str']))
    eval(model, dataloader , 2, 2, devicef, set, True, {'modelo': "{}_{}".format(checkpoint['str'],checkpoint['init']['version']), 'epoca': epoch, 'loss': '-', 'dataset': getDataset(dataloader)})

def generateMatrix_evals(model_load: str, set = 'valid',devicef = 1, tipo_head = '', filename = None):
    
    device = torch.device(devicef)

    checkpoint = torch.load(model_load, map_location=device)

    epoch = checkpoint['epoch']
    model = checkpoint['model']
    model.to(device)
    print('Modelo: {}, en la epoca: {} '.format(Path(model_load).name, epoch))
    eval_to_vector(model, 'JSONFiles/eyepacs_resam/eyepacs_', 1, 1, devicef, set, filename, Path(model_load).name, tipo_head)

def eval_to_vector(model, data: str, batch: int, workers: int, device: str, set: str, jsonfile: str, name ,tipo_head):

    dataloader = DataLoader(
        DrDataset(data + '{}.json'.format(set), set),
        batch_size=batch,
        num_workers=workers,shuffle=False
    )

    if tipo_head == 'att_h' and (name == 'hornet_0000_best_acc.pth' or name == 'hornet_0000_best_aa.pth' or name == 'hornet_0000_best_wk.pth'):
        model.head[8] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    elif tipo_head == 'att':
        model.cabf.fc_[8] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    elif tipo_head == 'att_h':
        model.att.fc_[8] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    elif tipo_head == 'att_c' and (name == 'convnext_small_0000_best_acc.pth' or name == 'convnext_small_0000_best_aa.pth' or name == 'convnext_small_0000_best_wk.pth'):
        model.attb[11] = torch.nn.Sequential(torch.nn.Softmax(dim=1))   
    elif tipo_head == 'att_c' :
        model.attb.fc_[8] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    elif tipo_head == 'att_i' and (name == 'internimage_0000_best_acc.pth' or name == 'internimage_0000_best_aa.pth'  or name == 'internimage_0000_best_wk.pth'):
        model.head[8] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    # # elif name == 'convnext_agus.pth':
    # #     model.classifier[10] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    # elif name == 'convnext_small_0000_best.pth':
    #     model.attb[11] = torch.nn.Sequential(torch.nn.Softmax(dim=1))
    # else:
    #     model.attb.fc_[8] = torch.nn.Sequential(torch.nn.Softmax(dim=1))

    

    model.eval()

    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    list_pred = Util.loadJSONFile(jsonfile)

    for _, batch in process_bar:
        image, label, f = batch
        label = label.squeeze()
        
        image = image.to(device)
        label = label.to(device)

        pred = model(image)

        for indv in pred.tolist():
            contains = getContainsDict(list_pred, f[0])
            if contains == -1:
                list_pred.append({
                    'filename': f[0],
                    'matrix': [indv],
                    'label': int(label)})
            else:
                list_pred[contains]['matrix'].append(indv)
    
    Util.savePredictionModels(jsonfile, list_pred)
    
def getContainsDict(array, filename):
    for i, obj in enumerate(array):
        if obj['filename'] == filename:
            return i
    return -1

def eval(model, data: str, batch: int, workers: int, device: str, set: str, test: bool = False, info : dict = {}, message= None):

    if test:
        dataloader = DataLoader(
            DrDataset(data + '{}.json'.format(set), set),
            batch_size=batch,
            num_workers=workers,
        )
    else:
        dataloader = DataLoader(
            DrDataset(data + '.json', set),
            batch_size=batch,
            num_workers=workers,
        )

    model.eval()
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    trues = []
    preds = []

    for _, batch in process_bar:
        image, label, f = batch
        label = label.squeeze()

        image = image.to(device)
        label = label.to(device)

        pred = model(image)

        try:
            trues.extend(label.tolist())
            preds.extend(torch.argmax(pred, dim=1).tolist())
        except:
            preds.append(int(torch.argmax(pred, dim=1)))
            trues.append(int(label))

        process_bar.set_description_str('Set: {}'.format(set), True)
        
    evals = MericsEvaluation(preds, trues, set)
    gs = GoogleService()
    info_r = [info['modelo'], info['epoca'], info['dataset'], info['loss'], set]
    info_r.extend(evals.getall())
    if message is not None:
        gs.insertRowToSheet([message])
    gs.insertRowToSheet(info_r)
    return evals.class_accuracy(), evals.accuracy(), evals.kappa()

    # if not save:
    #     cfm = confusion_matrix(trues, preds)
    #     if set == 'valid':
    #         img_total = [1253, 126, 895, 47, 182]

    #         acc_class = [float(cfm[0][0])/ img_total[0], float(cfm[1][1])/ img_total[1], 
    #                     float(cfm[2][2])/ img_total[2], float(cfm[3][3])/ img_total[3], 
    #                     float(cfm[4][4])/ img_total[4]]
        
    #     if set == 'train':
    #         img_total = [3133, 315, 2238, 118, 456]

    #         acc_class = [float(cfm[0][0])/ img_total[0], float(cfm[1][1])/ img_total[1], 
    #                     float(cfm[2][2])/ img_total[2], float(cfm[3][3])/ img_total[3], 
    #                     float(cfm[4][4])/ img_total[4]]

    #     return accuracy_score(trues, preds), acc_class

    # print(accuracy_score(trues, preds))

    # cfm = confusion_matrix(trues, preds).tolist()

    # print(cfm)

    # if set == 'valid':
    #     img_total = [1253, 126, 895, 47, 182]
    # if set == 'test':
    #     img_total = [1880, 189, 1344, 71, 275]
    
    # champ = 0.0
    # list_acc = {}
    # for i in range(5):
    #     res = float(cfm[i][i])/ img_total[i]
    #     list_acc[i] = res
    #     print('{}: {}'.format(i, res))

    #     champ += res

    # print('Campeon res: {:.2f}'.format((champ / 5) * 100))

    # return cfm, (champ / 5) * 100, list_acc

    
def getDataset(cadena):
    patron = r"/([^/]+)/"
    coincidencias = re.findall(patron, cadena)
    if coincidencias:
        return coincidencias[0]
    else:
        return None
