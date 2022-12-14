import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from data.drdataset import DrDataset
from models.resnet101 import resNet101Custom, resNet101Legacy, ResNet101AB
from models.convnext import convNextSmallCustom, convNextSmallegacy, ConvNextSmallAB
from models.resnet50 import ResNet50AB, resNet50Legacy
from models.convnext_small import convnext_small
from tqdm import tqdm
from utils.save_info import Util
from eval import eval
import os


def train(model_str, model_load, json_result, dump: str, data, epochs, lr, decay_lr,
          batch_t, batch_s, workers_t, workers_s, momentum, weigth_decay, devices, patience=3, set_lr=False):

    dataloader_train = DataLoader(
        DrDataset(data + 'train.json', 'train'),
        batch_size=batch_t,
        num_workers=workers_t,
        shuffle=True
    )

    device = torch.device(devices)
    classes = 5
    model = None

    if model_load is None:
        start_epoch = 0

        if model_str == 'resnet':
            model = resNet101Legacy(classes)

        if model_str == 'resnet50':
            model = resNet50Legacy(classes)

        if model_str == 'resnet50_abs':
            model = ResNet50AB()

        if model_str == 'resnet_custom':
            model = resNet101Custom(classes)

        if model_str == 'resnet_abs':
            model = ResNet101AB(classes=5, k=5)

        if model_str == 'resnet_abs_custom':
            model = ResNet101AB(classes=5, k=5, modo='custom')

        if model_str == 'convnext':
            model = convNextSmallegacy(classes)

        if model_str == 'convnext_custom':
            model = convNextSmallCustom(classes)

        if model_str == 'convnext_abs_original':
            model = ConvNextSmallAB(modo='original')

        if model_str == 'convnext_abs_custom':
            model = ConvNextSmallAB(modo='custom')

        if model_str == 'convnext_small':
            model = convnext_small(classes=5)

        optimizer = torch.optim.Adam(
            model.parameters(), lr, weight_decay=weigth_decay)

    else:
        checkpoint = torch.load(model_load, map_location=device)

        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

        if set_lr:
            for g in optimizer.param_groups:
                g['lr'] = lr

    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    data_eval = './JSONFiles/DDR/DDR_'

    best = 0.0
    best_dump = ''

    factor_lr = decay_lr
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', patience=patience, factor=factor_lr)

    for epoch in range(start_epoch, epochs):

        train_one_epoch(model, dataloader_train, optimizer,
                        criterion, epoch, device, json_result)

        Util.save_checkpoint(epoch, model, optimizer, dump, model_str)
        print('Evaluando....')

        acc, aps = eval(model, data_eval, batch_s,
                        workers_s, device, 'train', False)

        Util.saveInfoXepoch(os.path.dirname(json_result) +
                            '/info_train_{}.json'.format(model_str), epoch, acc, aps, 'train')

        acc, aps = eval(model, data_eval, batch_s,
                        workers_s, device, 'valid', False)

        Util.saveInfoXepoch(os.path.dirname(json_result) +
                            '/info_train_{}.json'.format(model_str), epoch, acc, aps, 'valid')

        if epoch > 30:
            print('Decaimiento')
            scheduler.step(acc)

        if best < acc:
            best = acc
            best_dump = os.path.dirname(json_result) + \
                '/{}_best.pth'.format(model_str)
            Util.save_checkpoint(epoch, model, optimizer, best_dump, model_str)


def train_one_epoch(model, dataloader, optimizer: torch.optim.Adam, criterion, epoch, device, json_result):

    model.train()
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    loss_total = 0.0

    for _, batch in process_bar:
        image, label, f = batch
        label = label.squeeze()

        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        pred = model(image)

        loss = criterion(pred, label)
        loss_total += float(loss)
        loss.backward()
        optimizer.step()
        process_bar.set_description_str(
            'Epoch {} : Loss: {:.3f}'.format(epoch + 1, float(loss)), True)

    Util.guardarLoss(json_result, loss_total/len(dataloader))
    print('Perdida promedio: {:.4f}'.format(loss_total/len(dataloader)))


def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale


def average(lst):
    return sum(lst) / len(lst)
