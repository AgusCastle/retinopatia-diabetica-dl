from models.convnext_small import convnext_small
from models.convnext import convNextSmallCustom , convNextSmallegacy
from models.hornet import hornet_small_gf_att, hornet_small_gf_agus
from models.intern_image import internImageSmallCAB, interImageSmallCustom
import torch
from thop import profile

def infoModels(modelo : str, device = 0):

    input_model = torch.randn((1, 3, 512, 512)).to(torch.device(0))

    model = torch.load(modelo, map_location=torch.device(device))['model']
    
    model.to(torch.device(device))
    print('Modelo -> {}'.format(modelo))
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
