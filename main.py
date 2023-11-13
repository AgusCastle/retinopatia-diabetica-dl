import argparse
import os
from sys import exit
from eval import eval, evalModelOneDataset, generateMatrix_evals
from train import train
from utils.save_info import Util
from utils.grad_cam import viewGradCam
from models.sparse import trainEval, evalSnf, SparseFusion
from utils.info_models import infoModels

MODELS = ['resnet50_abs', 
          'resnet50', 
          'convnext_custom', 
          'resnet', 
          'convnext', 
          'resnet_custom', 
          'resnet_abs_custom', 
          'resnet_abs', 
          'convnext_abs_original', 
          'convnext_abs_custom', 
          'convnext_small_', 
          'densenet121', 
          'resnext50',
          'hornet', 
          'internimage',
          'hornet_',
          'internimage_']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    # Acciones para el modelo
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='multi')
    parser.add_argument('--model', default=None)
    parser.add_argument('--att',  action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--gradcam', action='store_true', default=False)
    parser.add_argument('--matrix', action='store_true', default= False)
    parser.add_argument('--no_pretrain', action='store_true', default= False)
    parser.add_argument('--no_g', type=int, default=1)
    # python3.8 main.py --train --epochs 30 --lr 0.00001 --decay_lr 0.3 --batch 8 --workers 8 
    # --device 0 --patience 3 --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ 
    # --dump ./runs3/convnext_1001_1_eyepacs/convnext_1001.pth 
    # --json_result ./runs3/convnext_1001_1_eyepacs/results.json
    # --att --attn_block 1 0 0
    # Acciones para el dataset

    parser.add_argument('--gabor',type=int, default= 0)
    parser.add_argument('--data_update', action='store_true', default=False)
    parser.add_argument('--csv2json', action='store_true', default=False)
    parser.add_argument('--txt2json', action='store_true', default=False)

    # Hyperparametros
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--decay_lr', type=float, default=0.3)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weigth_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--set_lr', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--loss_sensitive', action='store_true', default=False)
    parser.add_argument('--loss_mode', type=int, default=1)
    parser.add_argument('--base_loss', type=str, default='ce')

    # Ubicaciones de archivos
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--dataloader_json', default=None)
    parser.add_argument('--json_file', default=None)
    parser.add_argument('--dump', default=None)
    parser.add_argument('--img_path', default=None)
    

    parser.add_argument('--txt', default=None)
    parser.add_argument('--path_src', default=None)
    parser.add_argument('--save_json', default=None)
    parser.add_argument('--set', default='train')

    parser.add_argument('--category', type=int, default=0)
    parser.add_argument('--attn_block', nargs='+', type=int)
    parser.add_argument('--msg', type=str, default='N/A')
    parser.add_argument('--general_info', action='store_true', default=False)
    # Entrenamiento SNF

    parser.add_argument('--snf_train', action='store_true', default=False)
    parser.add_argument('--snf_eval', action='store_true', default=False)

    args = parser.parse_args()

    if args.general_info:
        infoModels(modelo=args.model, device=args.device)
        exit()
    if args.snf_train:
        trainEval(args.dataloader_json ,args.lr, args.decay_lr, args.patience, args.epochs, args.batch, args.device,args.dump, loss_sensitive=args.loss_sensitive, loss_mode=args.loss_mode)
        exit()
        
    if args.snf_eval:
        evalSnf(args.dataloader_json, args.load_model, args.device)
        exit()
    # Visualizaciones GradCAM
    if args.gradcam:
        viewGradCam(args.load_model, args.category, args.img_path, args.device)

    # Evaluaciones para la SNF
    if args.matrix and args.load_model is not None:
        generateMatrix_evals(str(args.load_model), args.set, args.device, filename=args.msg)
        exit()

    # Evaluaciones para modelos individuales con un solo conjunto de datos
    if args.eval and args.load_model is not None and args.dataloader_json is not None:
        evalModelOneDataset(str(args.load_model), str(args.dataloader_json), args.set, args.device, args.msg)

    # Crea la carpeta donde se guardaran los pesos.
    if not os.path.exists('./runs'):
        os.makedirs('./runs', exist_ok=True)

    # Crea la carpeta donde se guardaran los Dataloaders
    if not os.path.exists('./JSONFiles'):
        os.makedirs('./JSONFiles', exist_ok=True)

    if args.data_update:
        Util.updateData(args.path_src, str(args.json_file))
        exit()

    # Opcion para generar un dataloader de un txt a json
    if args.txt2json:
        if args.txt is None or args.path_src is None:
            exit()
        Util.txt2json(args.txt, args.path_src, args.save_json, args.set)
        exit()

    # Entrenamiento de los modelos
    if args.train:

        if args.model is None or args.model not in MODELS:
            print('Elige un modelo a entrenar')
            exit()
            
        if args.dump is None:
            print('Elije donde guardar tus modelos')
            exit()
        else:
            if not os.path.exists(os.path.dirname(args.dump)):
                os.makedirs(os.path.dirname(args.dump))

        dump = args.dump

        if args.dataloader_json is None:
            print('Ingresa tu JSON de tu base de datos')
            exit()

        dataloader_json = args.dataloader_json

        epoch, lr, decay_lr, batch, workers, momentum, weigth_decay, device = args.epochs, args.lr, args.decay_lr, args.batch, args.workers, args.momentum, args.weigth_decay, args.device

        patience = args.patience
        set_lr = args.set_lr

        model_load = args.load_model

        train(args.model, model_load, dump,
              dataloader_json, epoch, lr, decay_lr, batch, 4,
              workers, 4, momentum, weigth_decay, device, patience, 
              set_lr, b_attn=args.attn_block, version=args.version, 
              mode=args.mode, att=args.att, no_pretrain=args.no_pretrain, 
              loss_sensitive=args.loss_sensitive, loss_mode=args.loss_mode, 
              base_loss=args.base_loss, gabor=args.gabor, no_g=args.no_g)
