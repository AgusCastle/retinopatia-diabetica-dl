# Tome los scripts y acomodelos si considera que los hyperparametros para from scratch estan mal muevalos jaja de todos modos voy a agregar en el sheet la relacion de los hyperparametros
# Solo hay que modificar los devices para que se tomen bien las gpus                            Esta ruta hay que checarla si solo agrego los json en jsonfiles hay que borrar                   

mkdir logs

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.01 --decay_lr 0.1 --patience 3 --att --attn_block 1 0 0 --dump ./runs/ConvNeXt_1001_9010_6/ConvNext_1001.pth --version 6 --no_pretrain > ./logs/log6.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.01 --decay_lr 0.1 --patience 3 --att --attn_block 1 0 1 --dump ./runs/ConvNeXt_1001_9010_7/ConvNext_1001.pth --version 7 --no_pretrain > ./logs/log7.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.001 --decay_lr 0.1 --patience 5 --att --attn_block 1 0 0 --dump ./runs/ConvNeXt_1001_9010_8/ConvNext_1001.pth --version 8 --no_pretrain > ./logs/log8.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.001 --decay_lr 0.1 --patience 5 --att --attn_block 1 0 1 --dump ./runs/ConvNeXt_1001_9010_9/ConvNext_1001.pth --version 9 --no_pretrain > ./logs/log9.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.1 --decay_lr 0.1 --patience 3 --att --attn_block 1 0 0 --dump ./runs/ConvNeXt_1001_9010_10/ConvNext_1001.pth --version 10 --no_pretrain > ./logs/log10.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.1 --decay_lr 0.1 --patience 3 --att --attn_block 1 0 1 --dump ./runs/ConvNeXt_1001_9010_11/ConvNext_1001.pth --version 11 --no_pretrain > ./logs/logs11.txt
