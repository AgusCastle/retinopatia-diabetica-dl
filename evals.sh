<<<<<<< HEAD
# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set valid --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_valid.json --device 0
# done;

# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set test --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_test.json --device 0
# done;

# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set train --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_train.json --device 0
# done;

python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_6/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 0 --load_model models_file/ConvNeXt_1011_13.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_7/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_7/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_8/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_8/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_9/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_9/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_10/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_10/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_11/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_11/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_12/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_12/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_13/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_13/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test
=======
# Tome los scripts y acomodelos si considera que los hyperparametros para from scratch estan mal muevalos jaja de todos modos voy a agregar en el sheet la relacion de los hyperparametros
# Solo hay que modificar los devices para que se tomen bien las gpus                            Esta ruta hay que checarla si solo agrego los json en jsonfiles hay que borrar                   

mkdir logs

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.02 --decay_lr 0.1 --patience 3 --att --attn_block 1 0 0 --dump ./runs/ConvNeXt_1001_9010_6/ConvNext_1001.pth --version 6 --no_pretrain > ./logs/log6.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.02 --decay_lr 0.1 --patience 3 --att --attn_block 1 0 1 --dump ./runs/ConvNeXt_1001_9010_7/ConvNext_1001.pth --version 7 --no_pretrain > ./logs/log7.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.002 --decay_lr 0.1 --patience 5 --att --attn_block 1 0 0 --dump ./runs/ConvNeXt_1001_9010_8/ConvNext_1001.pth --version 8 --no_pretrain > ./logs/log8.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.002 --decay_lr 0.1 --patience 5 --att --attn_block 1 0 1 --dump ./runs/ConvNeXt_1001_9010_9/ConvNext_1001.pth --version 9 --no_pretrain > ./logs/log9.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.2 --decay_lr 0.1 --patience 3xs --att --attn_block 1 0 0 --dump ./runs/ConvNeXt_1001_9010_10/ConvNext_1001.pth --version 10 --no_pretrain > ./logs/log10.txt

python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.2 --decay_lr 0.1 --patience 3 --att --attn_block 1 0 1 --dump ./runs/ConvNeXt_1001_9010_11/ConvNext_1001.pth --version 11 --no_pretrain > ./logs/logs11.txt
>>>>>>> 7da66516ef49946d458e3c585c149c4f939e37c2
