from models.convnext_small import convnext_small
from models.hornet import hornet_small_gf_att
from models.intern_image import internImageSmallCAB
import torch
from thop import profile

def infoModels(modelo : str, version = [0,0,0,0]):

    input_model = torch.randn((8, 3, 512, 512))

    if modelo == 'hornet':
        model = hornet_small_gf_att('', 5, version)
    elif modelo == 'internimage':
        model = internImageSmallCAB(5, version)
    else:
        model = convnext_small(5, True, version[-3:])
    
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
