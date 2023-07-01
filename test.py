import torch
from models.intern_image import interImageSmallCustom, internImageSmallCAB
from models.hornet import hornet_small_gf_att

modelo= hornet_small_gf_att('pretrain/hornet/hornet_small_gf.pth', 1000, [1,0,1,1])
print(modelo)
tensor_entrada = torch.randn(5, 3, 512, 512).to(torch.device('cuda:0'))  # Ejemplo: tensor de tamaño (1, 5)

# Realizar la inferencia
modelo.to(device=torch.device('cuda:0'))
modelo.eval()
resultado = modelo(tensor_entrada)

# Imprimir el resultado
# print(resultado.size())