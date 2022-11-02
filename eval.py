from statistics import mode
import torch
from torch.utils.data import DataLoader
from data.drdataset import DrDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


def bestEpoch(model_load: str):

    checkpoint = torch.load(model_load)

    epoch = checkpoint['epoch']
    model = checkpoint['model']
    device = torch.device(0)
    model.to(device)
    print(epoch)
    eval(model, 'JSONFiles/DDR/DDR_', 1, 1, 0, 'test', True)


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
        return accuracy_score(trues, preds), precision_score(trues, preds, average=None)

    print(accuracy_score(trues, preds))
    print(precision_score(trues, preds, average=None))
    print(confusion_matrix(trues, preds))
