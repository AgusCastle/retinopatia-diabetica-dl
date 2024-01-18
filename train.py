import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from data.drdataset import DrDataset
from models.resnet101 import resNet101Custom, resNet101Legacy, ResNet101AB
from models.convnext import convNextSmallCustom, convNextSmallegacy, ConvNextSmallAB
from models.resnet50 import resNet50Custom
from models.resnext50 import resNeXt50_agus
from models.densenet121 import denseNet121_agus
from models.convnext_small import convnext_small
from models.hornet import hornet_small_gf_agus, hornet_small_gf_att
from models.intern_image import interImageSmallCustom, internImageSmallCAB
from tqdm import tqdm
from utils.cost_lost_sensitive import CostSensitiveLoss, CostSensitiveRegularizedLoss
from utils.save_info import Util
from eval import eval
import os


def train(model_str, model_load, dump: str, data, epochs, lr, decay_lr,
          batch_t, batch_s, workers_t, workers_s, momentum, weigth_decay, devices, patience=3, set_lr=False, b_attn = [0, 0, 0], 
          version = 0, att = False, mode = 'multi', no_pretrain = False, loss_sensitive = False, loss_mode = 1, base_loss = 'ce', 
          gabor = 0, no_g = 1, warm_up = 15, save_per_epoch = []):

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

        if mode == 'bin':
            model = convnext_small(classes=2 , b_attn=b_attn)
        else:

            if model_str == 'resnet':
                model = resNet101Legacy(classes)

            if model_str == 'resnet50':
                model = resNet50Custom(classes)
            
            if model_str == 'resnext50':
                model = resNeXt50_agus(classes)
            
            if model_str == 'densenet121':
                model = denseNet121_agus(classes)

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

            if model_str == 'convnext_small_': # ConvNeXt_####
                model = convnext_small(classes=5 , b_attn=b_attn, pretrained= not no_pretrain, gabor=gabor, no_g=no_g)
            
            if model_str == 'hornet':
                model = hornet_small_gf_agus(pretrained_path='hornet/hornet_small_gf.pth',pretrained=True, classes=5)
            
            if model_str == 'internimage':
                model = interImageSmallCustom(5)

            if model_str == 'hornet_':
                model = hornet_small_gf_att(pretrained_path='pretrain/hornet/hornet_small_gf.pth',pretrained=True, classes=5, att=b_attn,  gabor=gabor, no_g=no_g)
            
            if model_str == 'internimage_':
                model = internImageSmallCAB(5,att=b_attn, device=devices, gabor=gabor, no_g=no_g)

        optimizer = torch.optim.Adam(
            model.parameters(), lr, weight_decay=weigth_decay)
        
        hypers = {
            'lr': lr,
            'wdecay': weigth_decay,
            'patience': patience,
            'version': version
        }

    else:
        checkpoint = torch.load(model_load, map_location=device)

        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        hypers = checkpoint['init']

        if set_lr:
            for g in optimizer.param_groups:
                g['lr'] = lr

    if loss_sensitive:
        criterion = CostSensitiveRegularizedLoss(5, reduction='mean', mode=loss_mode, base_loss=base_loss)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    data_eval = data

    best_aa = 0.0
    best_acc = 0.0
    best_wk = 0.0

    best_dump = ''

    factor_lr = decay_lr
    scheduler = ReduceLROnPlateau(
        optimizer, 'max', patience=patience, factor=factor_lr,verbose=True, min_lr= 1e-9)
    
    btt_name = ''
    if att:
        for i in b_attn:
            btt_name += str(i)
        model_str += btt_name

    for epoch in range(start_epoch, epochs):

        model, loss = train_one_epoch(model, dataloader_train, optimizer,
                        criterion, epoch, device, None)

        if epoch + 1 in save_per_epoch:
            Util.save_checkpoint(epoch, model, optimizer, dump + '{}.pth'.format(epoch + 1), model_str, info=hypers)
        
        Util.save_checkpoint(epoch, model, optimizer, dump, model_str, info=hypers)
        print('Evaluando....')

        #eval(model, data_eval, batch_s,
        #                workers_s, device, 'train', True, {'modelo':'{}_{}_{}'.format(model_str, btt_name, version), 'epoca': epoch, 'dataset': 'eyepacs', 'loss': loss})

        #Util.saveInfoXepoch(os.path.dirname(json_result) +
        #                    '/info_train_{}.json'.format(model_str), epoch, acc, aps, 'train')

        aa, acc, wk  = eval(model, data_eval, 2,
                        2, device, 'valid', True,  {'modelo': '{}_{}_{}'.format(model_str, btt_name, version), 'epoca': epoch, 'dataset': data.split('/')[2], 'loss': loss})

        # Util.saveInfoXepoch(os.path.dirname(json_result) +
         #                   '/info_train_{}.json'.format(model_str), epoch, acc, aps, 'valid')
        
        #eval(model, data_eval, batch_s,
        #               workers_s, device, 'test', False,  {'modelo': '{}_{}_{}'.format(model_str, btt_name, version) , 'epoca': epoch, 'dataset': 'messidor2', 'loss': '-'})

        if epoch > warm_up:
            scheduler.step(acc)

        if best_acc < acc:
            best_acc = acc
            best_dump = os.path.dirname(dump) + \
                '/{}_best_acc.pth'.format(model_str)
            Util.save_checkpoint(epoch, model, optimizer, best_dump, model_str, hypers)
        
        if best_aa < aa:
            best_aa = aa
            best_dump = os.path.dirname(dump) + \
                '/{}_best_aa.pth'.format(model_str)
            Util.save_checkpoint(epoch, model, optimizer, best_dump, model_str, hypers)

        if best_wk < wk:
            best_wk = wk
            best_dump = os.path.dirname(dump) + \
                '/{}_best_wk.pth'.format(model_str)
            Util.save_checkpoint(epoch, model, optimizer, best_dump, model_str, hypers)


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

    #Util.guardarLoss(json_result, loss_total/len(dataloader))
    print('Perdida promedio: {:.4f}'.format(loss_total/len(dataloader)))

    return model, loss_total/len(dataloader)


def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale


def average(lst):
    return sum(lst) / len(lst)
