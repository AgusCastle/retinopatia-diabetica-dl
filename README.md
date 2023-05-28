# Clasificacion de la retinopatia diabetica - ConvNeXt CabNET

## Requirements

- Python 3.8.6
- CUDA 10.2

## Ambientacion

- Creamos primero un entorno virtual en alguna carpeta y lo activamos.
    - ``` python3.8 -m venv retinopathy```
- Entramos a nuestro entorno virtual y ahora ejecutamos el requirements.txt
    - ``` python3.8 -m pip install -r requirements.txt```

## Datasets

Ya listo nuestro ambiente, vamos a crear los **json**, para tener una comparacion fiel de los entrenamientos vamos a ocupar los mismos que se generaron en el servidor.
- [Datasets EyePacs, Messidor2, DDR]() (Encuanto tenga acceso a las compus de la escuela subo esos archivos)

- Ya descargados, los acomodaremos en la carpeta JSONFiles en los directorios como estan en el zip.

- Despues se correra el siguiente comando para acomodar las paths correctas del servidor en cuestion. Esto se realizara una vez por cada JSON a actualizar.

    ``` python3.8 main.py --data_update --path_src <ruta nueva> --json_file <Ruta JSON a actualizar>```

## Entrenamiento

### Para entrenar se utiliza el siguiente comando

```
python3.8 main.py --train --epochs 10 --batch 8 --workers 8 --device 0 
--convnext_small_
--dataloader_json ./JSONFiles/DDR_
--lr 0.00002 decay_lr 0.3 --patience 3 --att --attn_block 0 0 1
--dump ./runs/ConvNeXt_0011/ConvNext_0011.pth
--version 1 
```

**_--dump_** es la ruta donde el modelo se va a guardar el modelo junto a su nombre

**_--dataloader_json_** este da la ruta de los json con los datos para entrenar, evaluar, etc. termininando con la base de datos que tenemos por ejemplo **_'./JSONFiles/DDR\_'_**

**_version_** este nos ayuda a diferenciarlo en el sheet mas que nada va ah ayudar para diferenciarlo de otros modelos.

Todos los modelos se guardan por epoca y se guarda el checkpoint con mejores resultados con respecto al Class Accuracy.

Se guardan en especifico modelo, nombre, optimizador, ultima epoca y informacion de hyperparametros.

En el caso que se requiera retomar el entrenamiento de un modelos se suguiere ocupar el mismo script pero agregando la siguiente linea:
 ``` --load_model <path del modelo a seguir entrenando>```

 Si se requiere entrenamiento sin los pesos de ImageNet agregar el siguiente parametro
 ```--no_pretrain```

## Evaluacion 
En dudoso caso que usted realice alguna evaluacion indivual de algun set se ocupa el siguiente script:

```python3.8 main.py --eval --device 0 --load_model <ruta del modelo> --dataloader_json <especificar la ruta al igual como si se fuera a entrenar> --set <valid,test,train>```

Los demas parametros los puede deducir facilmente XD

## Extras

Transformacion de data **_txt2json_**

Las clases seran guardadas en un **_json_** con el siguiente formato.

```
{
    filenames: [],
    labels: []
}
```

El nombre de estos JSON estan en el formato por ejemplo 'DDR_train.json'.

Para poder generarlos se debe correr el **_main.py_** con los siguientes argumentos:

```
python3.8 main.py
--txt2json
--txt /home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid.txt
--path_src /home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid
--save_json ./JSONFiles/DDR --set valid
```

**_--txt2json_** este aclara que se convertira el json

Donde **_--txt_** es la ruta del txt con la imagen y la label.

El txt debe tener el siguiente formato

```
nombredelaimagen etiqueta
Ejemplo:
imagen1.png 0
imagen2.png 1
.
.
.
```

**_--path_src_** aqui esta la ruta de las imagenes y siempre la ruta va al final sin el '/'

**_--save_json_** es la ruta donde se va guargar el formato debe ser **_./JSONFiles/{dataset}_** donde dataset se puede poner como en el ejemplo de arriba.

**_--set_** se define si es **valid** ó **test** ó **train**
