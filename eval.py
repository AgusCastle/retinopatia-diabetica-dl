from statistics import mode
import torch
from torch.utils.data import DataLoader
from data.drdataset import DrDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from utils.save_info import Util


def bestEpoch(model_load: str, set = 'valid',devicef = 1, filename = None):

    device = torch.device(devicef)

    checkpoint = torch.load(model_load, map_location=device)

    epoch = checkpoint['epoch']
    model = checkpoint['model']
    model.to(device)

    print('Ultima epoca: {}'.format(epoch))

    mc, acc, dicta = eval(model, 'JSONFiles/DDR/DDR_', 1, 1, devicef, set, True)
    if filename is not None:
        Util.guardarPrediction(filename, {
            'matriz' : mc,
            'acc'    : acc,
            'clases' : dicta
        })

def eval(model, data: str, batch: int, workers: int, device: str, set: str, save: bool = False):

    dataloader = DataLoader(
        DrDataset(data + '{}.json'.format(set), set),
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

        preds.append(int(torch.argmax(pred, dim=1)[0]))
        trues.append(int(label))
        process_bar.set_description_str('Set: {}'.format(set), True)

    if not save:
        cfm = confusion_matrix(trues, preds)
        if set == 'valid':
            img_total = [1253, 126, 895, 47, 182]

            acc_class = [float(cfm[0][0])/ img_total[0], float(cfm[1][1])/ img_total[1], 
                        float(cfm[2][2])/ img_total[2], float(cfm[3][3])/ img_total[3], 
                        float(cfm[4][4])/ img_total[4]]
        
        if set == 'train':
            img_total = [3133, 315, 2238, 118, 456]

            acc_class = [float(cfm[0][0])/ img_total[0], float(cfm[1][1])/ img_total[1], 
                        float(cfm[2][2])/ img_total[2], float(cfm[3][3])/ img_total[3], 
                        float(cfm[4][4])/ img_total[4]]

        return accuracy_score(trues, preds), acc_class

    print(accuracy_score(trues, preds))

    cfm = confusion_matrix(trues, preds).tolist()

    print(cfm)

    if set == 'valid':
        img_total = [1253, 126, 895, 47, 182]
    if set == 'test':
        img_total = [1880, 189, 1344, 71, 275]
    
    champ = 0.0
    list_acc = {}
    for i in range(5):
        res = float(cfm[i][i])/ img_total[i]
        list_acc[i] = res
        print('{}: {}'.format(i, res))

        champ += res

    print('Campeon res: {:.2f}'.format((champ / 5) * 100))

    return cfm, champ, list_acc

    
    
