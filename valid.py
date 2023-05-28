# import json 
# import random
# with open('JSONFiles/eyepacs_resam/eyepacs_test.json', 'r') as file:
#     data = json.load(file)

# paths = data['filenames']
# labels = data['labels']

# imagenes = {0:[], 1:[], 2:[],3:[],4:[]}

# for x, y in zip(paths, labels):
#     imagenes[y].append(x)

# total = [int(len(i) * .20) for i in imagenes.values()]

# print(total)

# tosave = {'filenames': [], 'labels': []}

# for i in range(5):
#     randoms = random.sample(range(0, len(imagenes[i])), total[i])

#     for j in randoms:
#         tosave['filenames'].append(imagenes[i][j])
#         tosave['labels'].append(i)

# print(len(tosave['filenames']))
# print(len(tosave['labels']))

# with open('./JSONFiles/eyepacs_cab/eyepacs_valid.json', 'w') as file:
#     json.dump(tosave, file)

import re

def getDataset(cadena):
    patron = r"/([^/]+)/"
    coincidencias = re.findall(patron, cadena)
    if coincidencias:
        return coincidencias[0]
    else:
        return None

print(getDataset('JSONFiles/messidor2/messidor2_'))