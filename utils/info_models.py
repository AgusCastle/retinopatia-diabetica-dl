from models.convnext_small import convnext_small
from models.convnext import convNextSmallCustom , convNextSmallegacy
from models.hornet import hornet_small_gf_att, hornet_small_gf_agus
from models.intern_image import internImageSmallCAB, interImageSmallCustom
import torch
from thop import profile

def infoModels(modelo : str, version = [0,0,0,0]):

    input_model = torch.randn((8, 3, 224, 224)).to(torch.device(0))

    if modelo == 'hornet':
        model = hornet_small_gf_agus('pretrain/hornet/hornet_small_gf.pth', 5)
    elif modelo == 'internimage':
        model = interImageSmallCustom(5)
    else:
        model = convNextSmallegacy(5)
    
    model.to(torch.device(0))
    print('Modelo -> {} version -> {}'.format(modelo, version))
    print('El numero de parametros: {}'.format(count_parameters(model)))

    print('Funcion para opbtener los FLOPS y parametros.')
    f, p = flops(model, input_model)
    print('FLOPS: {} -> Parametros: {}'.format(f, p))

def flops(model, input_model):
    #Touring cv
    flops, params = profile(model=model, inputs=(input_model,))

    return flops, params

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
