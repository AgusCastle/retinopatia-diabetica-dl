# import torch
# from models.intern_image import interImageSmallCustom, internImageSmallCAB
# from models.hornet import hornet_small_gf_att

# modelo= hornet_small_gf_att('pretrain/hornet/hornet_small_gf.pth', 1000, [1,0,1,1])
# print(modelo)
# tensor_entrada = torch.randn(5, 3, 512, 512).to(torch.device('cuda:0'))  # Ejemplo: tensor de tamaño (1, 5)

# # Realizar la inferencia
# modelo.to(device=torch.device('cuda:0'))
# modelo.eval()
# resultado = modelo(tensor_entrada)

# Imprimir el resultado
# print(resultado.size())

# Opcion de imagen para DR1
# /home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid/007-2551-100.jpg
# /home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid/007-2743-100.jpg
# /home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid/007-2685-100.jpg

# Opcion de imagen para DR3
# /home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid/007-5830-300.jpg

# Opcion de imagen para DR4
# /home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid/007-6860-400.jpg

import json 

def obtener_indice_maximo(arreglo):
    indice_maximo = max(range(len(arreglo)), key=lambda i: arreglo[i])
    return indice_maximo

with open('/Users/agustincastillo/Documents/DDR_valid.json', 'r') as file:
    data = json.load(file)

for img in data:
    if img['label'] not in (0, 2, 1, 3):
        maxs = [obtener_indice_maximo(img['matrix'][i]) for i in range(len(img['matrix']))]
        if maxs[1] != 4:
            print(img['filename'])
            print(maxs)
        