import json
import torch
import os
import csv
import pandas as pd
class Util():

    def __init__():
        super()

    def xls2json():

        data = {
            "filenames": [],
            "labels": []
        }

        for j, i in enumerate(['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34']):
            df = pd.read_excel('/home/bringascastle/Documentos/datasets-retina/Messidor/Annotation_Base{}.xls'.format(i), sheet_name='Feuil1')

            df = df[['Image name', 'Retinopathy grade']]

            for index, row in df.iterrows():
                if j in [0,1, 2, 3]:
                    data['filenames'].append( '/home/bringascastle/Documentos/datasets-retina/Messidor/Base1/'+ str(row['Image name']))
                if j in [4,5, 6, 7]:
                    data['filenames'].append( '/home/bringascastle/Documentos/datasets-retina/Messidor/Base2/'+ str(row['Image name']))
                if j in [8,9, 10, 11]:
                    data['filenames'].append( '/home/bringascastle/Documentos/datasets-retina/Messidor/Base3/'+ str(row['Image name']))

                data['labels'].append(int(row['Retinopathy grade']))
        
        print(len(data['filenames']))

        with open('JSONFiles/messidor/messidor _test.json', 'w') as file:
                json.dump(data, file)
    
    def xls2json_binary():

        data = {
            "filenames": [],
            "labels": []
        }

        for j, i in enumerate(['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34']):
            df = pd.read_excel('/home/bringascastle/Documentos/datasets-retina/Messidor/Annotation_Base{}.xls'.format(i), sheet_name='Feuil1')

            df = df[['Image name', 'Retinopathy grade']]

            for index, row in df.iterrows():
                if j in [0,1, 2, 3]:
                    data['filenames'].append( '/home/bringascastle/Documentos/datasets-retina/Messidor/Base1/'+ str(row['Image name']))
                if j in [4,5, 6, 7]:
                    data['filenames'].append( '/home/bringascastle/Documentos/datasets-retina/Messidor/Base2/'+ str(row['Image name']))
                if j in [8,9, 10, 11]:
                    data['filenames'].append( '/home/bringascastle/Documentos/datasets-retina/Messidor/Base3/'+ str(row['Image name']))

                if int(row['Retinopathy grade']) in [0 , 1]:
                    data['labels'].append(0)
                else:
                    data['labels'].append(1)

        print(len(data['filenames']))

        with open('JSONFiles/messidor/messidor_binary_train.json', 'w') as file:
                json.dump(data, file)

    def createInfoXepoch(filename):
        data = {
            'stats': []
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    def saveInfoXepoch(filename, epoch, acc, aps, data):

        with open(filename, 'r') as file:
            datos = json.load(file)

        data = {
            "set": data,
            "epoch": epoch,
            "acc": acc,
            "none": aps[0],
            "mild": aps[1],
            "moderate": aps[2],
            "severe": aps[3],
            "pro_dr": aps[4]
        }

        datos['stats'].append(data)

        with open(filename, 'w') as file:
            json.dump(datos, file)

    def txt2json(file, path_src, path_to, set):

        data = {
            "filenames": [],
            "labels": []
        }

        with open(file, 'r') as arch:
            for info in arch:
                if int(info.split(' ')[1].rstrip('\n')) != 5:
                    data['filenames'].append(
                        path_src + '/' + info.split(' ')[0])
                    data['labels'].append(int(info.split(' ')[1].rstrip('\n')))

        with open('./JSONFiles/{}_{}.json'.format(path_to, set), 'w') as file:
            json.dump(data, file)

        print('Se ha generado el JSON en {}'.format(path_to))

    def csv2json(path_csv, path_src, path_to, columna_i, columna_g, ext=None):

        data = {
            "filenames": [],
            "labels": []
        }

        with open(path_csv, 'r') as file:
            datos = csv.reader(file, delimiter=',')
        
            next(datos, None)

            for fila in datos:
                img = fila[columna_i]
                # if img in ['20060411_58550_0200_PP.png', 'IM002385.jpg', 'IM004176.jpg', 'IM003718.jpg']:
                #    continue
                grad = fila[columna_g]

                if ext is None:
                    data['filenames'].append(str(path_src + '/' + img))
                    data['labels'].append(int(grad))
                else:
                    if not os.path.exists(str(path_src + '/' + img + ext)):
                        print('No existe: {}'.format(str(path_src + '/' + img + ext)))
                    data['filenames'].append(str(path_src + '/' + img + ext))
                    data['labels'].append(int(grad))

            with open(path_to, 'w') as file:
                json.dump(data, file)

            print('Se ha generado el JSON en: {}'.format(path_to))

    def save_checkpoint(epoch, model, optimizer, filename, model_str, info):
        state = {
            'str': model_str,
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'init': info
        }

        torch.save(state, filename)

    def clean_loss(filename):
        with open(filename, 'r') as file:
            data = json.load(file)

        data['loss'].clear()

        with open(filename, 'w') as file:
            json.dump(data, file)

    def clean_preds(filename):
        with open(filename, 'r') as file:
            data = json.load(file)

        data['predictions'].clear()

        with open(filename, 'w') as file:
            json.dump(data, file)

    def generarJSON(filename):

        data = {
            "loss": [],
            "predictions": []
        }

        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    def guardarLoss(filename, loss):

        with open(filename, 'r') as file:
            data = json.load(file)

        data['loss'].append(loss)

        with open(filename, 'w') as file:
            json.dump(data, file)

    def guardarPrediction(filename, datas):

        with open(filename, 'r') as file:
            data = json.load(file)

        data['predictions'].append(datas)

        with open(filename, 'w') as file:
            json.dump(data, file)
    
    def loadJSONFile(filename):

        if not os.path.exists(filename):
            with open(filename, 'w') as archivo:
                json.dump([], archivo)

        with open(filename, 'r') as file:
            data = json.load(file)
        
        return data

    def savePredictionModels(filename, datas):
        with open(filename, 'w') as file:
            json.dump(datas, file)
    
    def updateData(path, jsonfile):
        with open(jsonfile, 'r') as file:
            data = json.load(file)
        
        for i in range(len(data['filenames']) - 1):
            data['filenames'][i] = os.path.join(path ,str(data['filenames'][i]).split('/')[-1])

        with open(jsonfile, 'w') as file:
            json.dump(data, file)
#Util.csv2json('/home/bringascastle/Documentos/datasets-retina/kaggle/testLabels.csv',
#                '/home/bringascastle/Documentos/datasets-retina/kaggle/test', 'JSONFiles/eyepacs_resam/eyepacs_test.json',0, 1, '.jpeg')

# Util.xls2json_binary()