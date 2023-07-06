#for i in runs6/ConvNeXt_0000_02/convnext_small_0000_best.pth runs6/ConvNeXt_0001_01/convnext_small_0001_best.pth runs6/ConvNeXt_0011_01/convnext_small_0011_best.pth runs6/ConvNeXt_0101_01/convnext_small_0101_best.pth runs6/ConvNeXt_0111_02/convnext_small_0111_best.pth runs6/ConvNeXt_1001_01/convnext_small_1001_best.pth runs6/ConvNeXt_1011_02/convnext_small_1011_best.pth runs6/ConvNeXt_1101_01/convnext_small_1101_best.pth runs6/ConvNeXt_1111_02/convnext_small_1111_best.pth; do
     #python3.8 main.py --matrix --set test --load_model runs_models/$i --msg JSONFiles/eyepacs_log_m/messidor2_test.json --device 0
#done;

# for i in runs6/ConvNeXt_0000_02/convnext_small_0000_best.pth runs6/ConvNeXt_0001_01/convnext_small_0001_best.pth runs6/ConvNeXt_0011_01/convnext_small_0011_best.pth runs6/ConvNeXt_0101_01/convnext_small_0101_best.pth runs6/ConvNeXt_0111_02/convnext_small_0111_best.pth runs6/ConvNeXt_1001_01/convnext_small_1001_best.pth runs6/ConvNeXt_1011_02/convnext_small_1011_best.pth runs6/ConvNeXt_1101_01/convnext_small_1101_best.pth runs6/ConvNeXt_1111_02/convnext_small_1111_best.pth; do
#      python3.8 main.py --matrix --set test --load_model $i --msg JSONFiles/eyepacs_log_m/eyepacs_test.json --device 0
# done; 

# for i in runs6/ConvNeXt_0000_02/convnext_small_0000_best.pth runs6/ConvNeXt_0001_01/convnext_small_0001_best.pth runs6/ConvNeXt_0011_01/convnext_small_0011_best.pth runs6/ConvNeXt_0101_01/convnext_small_0101_best.pth runs6/ConvNeXt_0111_02/convnext_small_0111_best.pth runs6/ConvNeXt_1001_01/convnext_small_1001_best.pth runs6/ConvNeXt_1011_02/convnext_small_1011_best.pth runs6/ConvNeXt_1101_01/convnext_small_1101_best.pth runs6/ConvNeXt_1111_02/convnext_small_1111_best.pth; do
#      python3.8 main.py --matrix --set valid --load_model $i --msg JSONFiles/eyepacs_log_m/eyepacs_valid.json --device 0
# done;

# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set valid --load_model runs_models/runs2/$i/$i.pth --msg JSONFiles/DDR_M/DDR_valid.json --device 0
# done;

# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set test --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_test.json --device 0
# done;

# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set train --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_train.json --device 0
# done;
#python3.8 main.py --train --epochs 30 --lr 0.000007 --decay_lr 0.2 --batch 8 --workers 8 --device 1 --patience 3 --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --dump ./runs4/ConvNeXt_0000_9010_22/ConvNeXt_0000.pth --version 22 --model convnext_custom --mode multi
# Entrenamiento GPU 1

#python3.8 main.py --train --epochs 30 --lr 0.000005 --decay_lr 0.6 --batch 8 --workers 8 --device 0 --patience 3 --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --dump ./runs4/ConvNeXt_0000_9010_23/ConvNeXt_0000.pth --version 23 --model convnext_custom --mode multi --load_model ./runs4/ConvNeXt_0000_9010_23/ConvNeXt_0000.pth
#for i in runs6/ConvNeXt_0000_02/convnext_small_0000_best.pth runs6/ConvNeXt_0001_01/convnext_small_0001_best.pth runs6/ConvNeXt_0011_01/convnext_small_0011_best.pth runs6/ConvNeXt_0101_01/convnext_small_0101_best.pth runs6/ConvNeXt_0111_02/convnext_small_0111_best.pth runs6/ConvNeXt_1001_01/convnext_small_1001_best.pth runs6/ConvNeXt_1011_02/convnext_small_1011_best.pth runs6/ConvNeXt_1101_01/convnext_small_1101_best.pth runs6/ConvNeXt_1111_02/convnext_small_1111_best.pth; do
#     python3.8 main.py --matrix --set test --load_model $i --msg JSONFiles/eyepacs_M/eyepacs_test.json --device 1
#done;


# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set test --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_test.json --device 0
# done;

# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set train --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_train.json --device 0
# done;

# for i in 01 02; do
#     for s in 0001 0011 0101 0111 1001 1011 1101 1111; do
#         #python3.8 main.py --eval --device 1 --load_model runs6/ConvNeXt_"$s"_"$i"/convnext_small_"$s"_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test --msg eval_conf_"$i"_"$s"_eyepacs
#         python3.8 main.py --eval --device 0 --load_model runs6/ConvNeXt_"$s"_"$i"/convnext_small_"$s"_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test --msg eval_conf_"$i"_"$s"_messidor
#     done;
# done;
# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_6/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
# python3.8 main.py --eval --device 0 --load_model models_file/ConvNeXt_1011_13.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_7/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_7/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_8/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_8/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_9/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_9/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_10/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_10/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_11/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_11/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_12/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1001_9010_12/convnext_small_1001_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

# python3.8 main.py --eval --device 0 --load_model runs4/Hornet/hornet_best.pth --dataloader_json JSONFiles/DDR/DDR_ --set test
# python3.8 main.py --eval --device 0 --load_model runs5/ConvNeXt_1011_9010_13/convnext_small_1011_best.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

# # python3.8 main.py --train --epochs 30 --lr 0.000005 --decay_lr 0.6 --batch 8 --workers 8 --device 0 --patience 3 --dataloader_json ./JSONFiles/eyepacs/eyepacs_ --dump ./runs/ConvNeXt_0001_9010_24/ConvNeXt_1001.pth --version 24 --model convnext_small_ --attn_block 0 0 0 --mode multi --att
# python3.8 main.py --snf_train --lr 1.2 --decay_lr 0.6 --patience 100 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_12.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_19.pth

# python3.8 main.py --snf_train --lr 0.3 --decay_lr 0.9 --patience 300 --epochs 800 --batch 256 --device 0 --dump ./runSNF/SNF_5.pth 
# # python3.8 main.py --snf_train --lr 1.3 --decay_lr 0.9 --patience 200 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_7.pth 
# python3.8 main.py --snf_train --lr 2.2 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runSNF/SNF_8.pth 
# python3.8 main.py --snf_train --lr 2.1 --decay_lr 0.8 --patience 80 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_9.pth 
# python3.8 main.py --snf_train --lr 0.9 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runSNF/SNF_10.pth 
# python3.8 main.py --snf_train --lr 0.9 --decay_lr 0.7 --patience 80 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_11.pth 
# python3.8 main.py --snf_train --lr 1.5 --decay_lr 0.6 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runSNF/SNF_12.pth

# python3.8 main.py --snf_train --lr 1.5 --decay_lr 0.6 --patience 80 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_13.pth
# python3.8 main.py --snf_train --lr 1.2 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runSNF/SNF_14.pth
# python3.8 main.py --snf_train --lr 1.2 --decay_lr 0.7 --patience 90 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_15.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runSNF/SNF_16.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_17.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.9 --patience 80 --epochs 400 --batch 128 --device 0 --dump ./runSNF/SNF_18.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_19.pth

python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 0 0 1 --dump ./runs_models/runs7/internimage_0001/internimage_0001.pth --version 1
python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 0 1 1 --dump ./runs_models/runs7/internimage_0011/internimage_0011.pth --version 2
python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 1 0 1 --dump ./runs_models/runs7/internimage_0101/internimage_0101.pth --version 3
python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 1 1 1 --dump ./runs_models/runs7/internimage_0111/internimage_0111.pth --version 4
python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 0 0 1 --dump ./runs_models/runs7/internimage_1001/internimage_1001.pth --version 5
python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 0 1 1 --dump ./runs_models/runs7/internimage_1011/internimage_1011.pth --version 6
python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs7/internimage_1101/internimage_1101.pth --version 7
python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 1 1 1 --dump ./runs_models/runs7/internimage_1111/internimage_1111.pth --version 8 --load_model ./runs_models/runs7/internimage_1111/internimage_1111.pth

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 0 0 1 --dump ./runs_models/runs7/hornet_0001/hornet_0001.pth --version 1
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 0 1 1 --dump ./runs_models/runs7/hornet_0011/hornet_0011.pth --version 2
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 1 0 1 --dump ./runs_models/runs7/hornet_0101/hornet_0101.pth --version 3
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 1 1 1 --dump ./runs_models/runs7/hornet_0111/hornet_0111.pth --version 4
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 0 0 1 --dump ./runs_models/runs7/hornet_1001/hornet_1001.pth --version 5
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 0 1 1 --dump ./runs_models/runs7/hornet_1011/hornet_1011.pth --version 6
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs7/hornet_1101/hornet_1101.pth --version 7
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 1 1 1 --dump ./runs_models/runs7/hornet_1111/hornet_1111.pth --version 8