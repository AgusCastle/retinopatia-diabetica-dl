# source /home/bringascastle/Documentos/venvs/drpredict/bin/activate
# cd /home/bringascastle/Documentos/repos/retinopatia-diabetica-dl/
#for i in runs6/ConvNeXt_0000_02/convnext_small_0000_best.pth runs6/ConvNeXt_0001_01/convnext_small_0001_best.pth runs6/ConvNeXt_0011_01/convnext_small_0011_best.pth runs6/ConvNeXt_0101_01/convnext_small_0101_best.pth runs6/ConvNeXt_0111_02/convnext_small_0111_best.pth runs6/ConvNeXt_1001_01/convnext_small_1001_best.pth runs6/ConvNeXt_1011_02/convnext_small_1011_best.pth runs6/ConvNeXt_1101_01/convnext_small_1101_best.pth runs6/ConvNeXt_1111_02/convnext_small_1111_best.pth; do
     #python3.8 main.py --matrix --set test --load_model runs_models/$i --msg JSONFiles/eyepacs_log_m/messidor2_test.json --device 0
#done;

# for s in 0001 0011 0101 0111 1001 1011 1101; do
#      python3.8 main.py --eval --load_model runs_models/runs7/internimage_"$s"/internimage_"$s"_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0 
#      python3.8 main.py --eval --load_model runs_models/runs7/hornet_"$s"/hornet_"$s"_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
#      done;

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
# python3.8 main.py --snf_train --lr 1.3 --decay_lr 0.9 --patience 200 --epochs 400 --batch 256 --device 0 --dump ./runSNF/SNF_7.pth 
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

# #python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.8 --patience 3 --att --attn_block 0 0 0 0 --dump ./runs_models/runs7/internimage_0000_2/internimage_0000.pth --version 8
# python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.8 --patience 3 --att --attn_block 0 0 0 1 --dump ./runs_models/runs7/internimage_0001_2/internimage_0001.pth --version 15
# python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.8 --patience 3 --att --attn_block 0 0 1 1 --dump ./runs_models/runs7/internimage_0011_2/internimage_0011.pth --version 10
# python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.8 --patience 3 --att --attn_block 0 1 0 1 --dump ./runs_models/runs7/internimage_0101_2/internimage_0101.pth --version 11
# python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.8 --patience 3 --att --attn_block 0 1 1 1 --dump ./runs_models/runs7/internimage_0111_2/internimage_0111.pth --version 12
# python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.8 --patience 3 --att --attn_block 1 0 0 1 --dump ./runs_models/runs7/internimage_1001_2/internimage_1001.pth --version 13
# python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.8 --patience 3 --att --attn_block 1 0 1 1 --dump ./runs_models/runs7/internimage_1011_2/internimage_1011.pth --version 14
# #python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.2 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs7/internimage_1101_2/internimage_1101.pth --version 15
# python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.8 --patience 3 --att --attn_block 1 1 1 1 --dump ./runs_models/runs7/internimage_1111_2/internimage_1111.pth --version 16 

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs11/internimage_1101/internimage_1101.pth --version 150

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs10/internimage_1101_1/internimage_1101.pth --version 1 --loss_sensitive --loss_mode 3 --base_loss gls
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs10/internimage_1101_2/internimage_1101.pth --version 2 --loss_sensitive --loss_mode 3 --base_loss focal_loss
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs10/internimage_1101_3/internimage_1101.pth --version 3 --loss_sensitive --loss_mode 3 --base_loss ce
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs8/internimage_0011_2/internimage_0011.pth --version 40
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000009 --decay_lr 0.9 --patience 3 --att --attn_block 0 0 1 1 --dump ./runs_models/runs8/internimage_0011_3/internimage_0011.pth --version 50
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000003 --decay_lr 0.9 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs8/internimage_0011_4/internimage_0011.pth --version 60

# python3.8 main.py --eval --load_model runs_models/runs8/internimage_0011_1/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# python3.8 main.py --eval --load_model runs_models/runs8/internimage_0011_1/internimage_0011.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# python3.8 main.py --eval --load_model runs_models/runs8/internimage_0011_2/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# python3.8 main.py --eval --load_model runs_models/runs8/internimage_0011_2/internimage_0011.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# python3.8 main.py --eval --load_model runs_models/runs8/internimage_0011_3/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# python3.8 main.py --eval --load_model runs_models/runs8/internimage_0011_3/internimage_0011.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# python3.8 main.py --eval --load_model runs_models/runs8/internimage_0011_4/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# python3.8 main.py --eval --load_model runs_models/runs8/internimage_0011_4/internimage_0011.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 0 0 1 --dump ./runs_models/runs7/hornet_0001/hornet_0001.pth --version 1
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 0 1 1 --dump ./runs_models/runs7/hornet_0011/hornet_0011.pth --version 2
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 1 0 1 --dump ./runs_models/runs7/hornet_0101/hornet_0101.pth --version 3
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 0 1 1 1 --dump ./runs_models/runs7/hornet_0111/hornet_0111.pth --version 4
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 0 0 1 --dump ./runs_models/runs7/hornet_1001/hornet_1001.pth --version 5
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 0 1 1 --dump ./runs_models/runs7/hornet_1011/hornet_1011.pth --version 6
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs7/hornet_1101/hornet_1101.pth --version 7
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.2 --patience 3 --att --attn_block 1 1 1 1 --dump ./runs_models/runs7/hornet_1111/hornet_1111.pth --version 8

# InternImage 0011 es mejor 1101 por lo tanto los resultados se realizaron con 0011
# Ver si la funcion softmax es mejor que

# 1 Funcion de activacion con SoftMax y CE
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011/internimage_0011.pth --version 160

# 2 LogSoftmax es el ganador : Probaremos M y Mopt por 30 epocas
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m/internimage_0011.pth --version 170 --loss_sensitive --loss_mode 1 --base_loss ce
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth/internimage_0011.pth --version 180 --loss_sensitive --loss_mode 2 --base_loss ce
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2/internimage_0011.pth --version 190 --loss_sensitive --loss_mode 3 --base_loss ce
# 3 comparar las funciones de perdida gano los mode 3

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_gls/internimage_0011.pth --version 200 --loss_sensitive --loss_mode 3 --base_loss gls

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_fl/internimage_0011.pth --version 210 --loss_sensitive --loss_mode 3 --base_loss focal_loss

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_ce/internimage_0011.pth --version 220 --loss_sensitive --loss_mode 3 --base_loss ce

# evaluaciones de tests

# # 1
# python3.8 main.py --eval --load_model ./runs_models/runs11/internimage_0011/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # 2
# python3.8 main.py --eval --load_model ./runs_models/runs11/internimage_0011_m/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# python3.8 main.py --eval --load_model ./runs_models/runs11/internimage_0011_m_opth/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# python3.8 main.py --eval --load_model ./runs_models/runs11/internimage_0011_m_opth_2/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # 3
# python3.8 main.py --eval --load_model ./runs_models/runs11/internimage_0011_m_opth_2_gls/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# python3.8 main.py --eval --load_model ./runs_models/runs11/internimage_0011_m_opth_2_fl/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# python3.8 main.py --eval --load_model ./runs_models/runs11/internimage_0011_m_opth_2_ce/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# 4 prueba con M por 60 epocas conf 220

#python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_ce_60/internimage_0011.pth --version 230 --loss_sensitive --loss_mode 1 --base_loss ce

# Prueba de gabor con el de Petonic o como se llame
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_ce_gabor/internimage_0011.pth --version 300 --loss_sensitive --loss_mode 3 --base_loss ce
# Prueba de gabor con Pentopnic ICG antes de gab
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_ce_gabor_310/internimage_0011.pth --version 310 --loss_sensitive --loss_mode 3 --base_loss ce
# Prueba de gabor despues de gab
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_ce_gabor_320/internimage_0011.pth --version 320 --loss_sensitive --loss_mode 3 --base_loss ce
# Prueba gabor antes de gab
# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_ce_gabor_330/internimage_0011.pth --version 330 --loss_sensitive --loss_mode 3 --base_loss ce
# Prueba gabor antes de gab por 60 epocas
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 1  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_60_ce_gabor_340/internimage_0011.pth --version 340 --loss_sensitive --loss_mode 3 --base_loss ce
# prueba d4 gabor despues de gab 60 epocas
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_60_ce_gabor_350/internimage_0011.pth --version 350 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# Evaluacion despues de gabor
# python3.8 main.py --eval --load_model ./runs_models/runs11/internimage_0011_m_opth_2_60_ce_gabor_350/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# Prueba gabor antes de gab por 60 epocas

#python3.8 main.py --eval --load_model runs_models/runs10/internimage_1101_1/internimage_1101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

#python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs11/internimage_0011_m_opth_2_60_ce_gabor_340/internimage_0011.pth --load_model ./runs_models/runs11/internimage_0011_m_opth_2_60_ce_gabor_340/internimage_0011.pth --version 340 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 1 --no_g 1

# Entrenamientos de los modelos
# echo 'Entrenamientos ... 0001'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1 --dump ./runs_models/runs12/internimage_0001_m_opth_2_60_ce_agabor_1/internimage_0001.pth --version 1 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs12/internimage_0001_m_opth_2_60_ce_agabor_1/internimage_0001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 0011'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs12/internimage_0011_m_opth_2_60_ce_agabor_2/internimage_0011.pth --version 2 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs12/internimage_0011_m_opth_2_60_ce_agabor_2/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 0101'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 0 1 --dump ./runs_models/runs12/internimage_0101_m_opth_2_60_ce_agabor_3/internimage_0101.pth --version 3 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs12/internimage_0101_m_opth_2_60_ce_agabor_3/internimage_0101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 0111'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 1 1 --dump ./runs_models/runs12/internimage_0111_m_opth_2_60_ce_agabor_4/internimage_0111.pth --version 4 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs12/internimage_0111_m_opth_2_60_ce_agabor_4/internimage_0111_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 1001'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 0 1 --dump ./runs_models/runs12/internimage_1001_m_opth_2_60_ce_agabor_5/internimage_1001.pth --version 5 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs12/internimage_1001_m_opth_2_60_ce_agabor_5/internimage_1001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 1011'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 1 1 --dump ./runs_models/runs12/internimage_1011_m_opth_2_60_ce_agabor_6/internimage_1011.pth --version 6 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs12/internimage_1011_m_opth_2_60_ce_agabor_6/internimage_1011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 1101'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs12/internimage_1101_m_opth_2_60_ce_agabor_7/internimage_1101.pth --version 7 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs12/internimage_1101_m_opth_2_60_ce_agabor_7/internimage_1101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 1111'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 1 1 --dump ./runs_models/runs12/internimage_1111_m_opth_2_60_ce_agabor_8/internimage_1111.pth --version 8 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs12/internimage_1111_m_opth_2_60_ce_agabor_8/internimage_1111_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamiento 0000'
#python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/runs12/internimage_0000_m_opth_2_60_ce_0/internimage_0000.pth --version 0 --loss_sensitive --loss_mode 3 --base_loss ce --load_model ./runs_models/runs12/internimage_0000_m_opth_2_60_ce_0/internimage_0000.pth
#python3.8 main.py --eval --load_model ./runs_models/runs12/internimage_0000_m_opth_2_60_ce_0/internimage0000_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0 

# Evaluaciones para obtener los softmax scores

# for i in runs_models/runs12/internimage_0000_m_opth_2_60_ce_0/internimage0000_best.pth runs_models/runs12/internimage_0001_m_opth_2_60_ce_agabor_1/internimage_0001_best.pth runs_models/runs12/internimage_0011_m_opth_2_60_ce_agabor_2/internimage_0011_best.pth runs_models/runs12/internimage_0101_m_opth_2_60_ce_agabor_3/internimage_0101_best.pth runs_models/runs12/internimage_0111_m_opth_2_60_ce_agabor_4/internimage_0111_best.pth runs_models/runs12/internimage_1001_m_opth_2_60_ce_agabor_5/internimage_1001_best.pth runs_models/runs12/internimage_1011_m_opth_2_60_ce_agabor_6/internimage_1011_best.pth runs_models/runs12/internimage_1101_m_opth_2_60_ce_agabor_7/internimage_1101_best.pth runs_models/runs12/internimage_1111_m_opth_2_60_ce_agabor_8/internimage_1111_best.pth; do
#     python3.8 main.py --matrix --set test --load_model $i --msg JSONFiles/DDR_M_I/DDR_test.json --device 0
# done;

# for i in runs_models/runs12/internimage_0000_m_opth_2_60_ce_0/internimage0000_best.pth runs_models/runs12/internimage_0001_m_opth_2_60_ce_agabor_1/internimage_0001_best.pth runs_models/runs12/internimage_0011_m_opth_2_60_ce_agabor_2/internimage_0011_best.pth runs_models/runs12/internimage_0101_m_opth_2_60_ce_agabor_3/internimage_0101_best.pth runs_models/runs12/internimage_0111_m_opth_2_60_ce_agabor_4/internimage_0111_best.pth runs_models/runs12/internimage_1001_m_opth_2_60_ce_agabor_5/internimage_1001_best.pth runs_models/runs12/internimage_1011_m_opth_2_60_ce_agabor_6/internimage_1011_best.pth runs_models/runs12/internimage_1101_m_opth_2_60_ce_agabor_7/internimage_1101_best.pth runs_models/runs12/internimage_1111_m_opth_2_60_ce_agabor_8/internimage_1111_best.pth; do
#     python3.8 main.py --matrix --set valid --load_model $i --msg JSONFiles/DDR_M_I/DDR_valid.json --device 0
# done;

# for i in runs_models/runs12/internimage_0000_m_opth_2_60_ce_0/internimage0000_best.pth runs_models/runs12/internimage_0001_m_opth_2_60_ce_agabor_1/internimage_0001_best.pth runs_models/runs12/internimage_0011_m_opth_2_60_ce_agabor_2/internimage_0011_best.pth runs_models/runs12/internimage_0101_m_opth_2_60_ce_agabor_3/internimage_0101_best.pth runs_models/runs12/internimage_0111_m_opth_2_60_ce_agabor_4/internimage_0111_best.pth runs_models/runs12/internimage_1001_m_opth_2_60_ce_agabor_5/internimage_1001_best.pth runs_models/runs12/internimage_1011_m_opth_2_60_ce_agabor_6/internimage_1011_best.pth runs_models/runs12/internimage_1101_m_opth_2_60_ce_agabor_7/internimage_1101_best.pth runs_models/runs12/internimage_1111_m_opth_2_60_ce_agabor_8/internimage_1111_best.pth; do
#     python3.8 main.py --matrix --set train --load_model $i --msg JSONFiles/DDR_M_I/DDR_train.json --device 0
# done;


#La SNF va a ssalir mal pero yo sigo al pie de la letra 

# python3.8 main.py --snf_train --lr 1.2 --decay_lr 0.6 --patience 100 --epochs 400 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_12_2.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_19_2.pth

# python3.8 main.py --snf_train --lr 0.3 --decay_lr 0.9 --patience 300 --epochs 800 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_5.pth 
# python3.8 main.py --snf_train --lr 1.3 --decay_lr 0.9 --patience 200 --epochs 400 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_7.pth 
# python3.8 main.py --snf_train --lr 2.2 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runs_models/runSNF2/SNF_8.pth 
# python3.8 main.py --snf_train --lr 2.1 --decay_lr 0.8 --patience 80 --epochs 400 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_9.pth 
# python3.8 main.py --snf_train --lr 0.9 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runs_models/runSNF2/SNF_10.pth 
# python3.8 main.py --snf_train --lr 0.9 --decay_lr 0.7 --patience 80 --epochs 400 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_11.pth 
# python3.8 main.py --snf_train --lr 1.5 --decay_lr 0.6 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runs_models/runSNF2/SNF_12.pth

# python3.8 main.py --snf_train --lr 1.5 --decay_lr 0.6 --patience 80 --epochs 400 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_13.pth
# python3.8 main.py --snf_train --lr 1.2 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runs_models/runSNF2/SNF_14.pth
# python3.8 main.py --snf_train --lr 1.2 --decay_lr 0.7 --patience 90 --epochs 400 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_15.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 0 --dump ./runs_models/runSNF2/SNF_16.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_17.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.9 --patience 80 --epochs 400 --batch 128 --device 0 --dump ./runs_models/runSNF2/SNF_18.pth
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 0 --dump ./runs_models/runSNF2/SNF_19.pth

# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_12_2.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_19_2.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_5.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_7.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_8.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_9.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_10.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_11.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_12.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_13.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_14.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_15.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_16.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_17.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_18.pthbest.pth --device 0
# python3.8 main.py --snf_eval --load_model runs_models/runSNF2/SNF_19.pthbest.pth --device 0

# Entrenamientos de los modelos
# echo 'Entrenamientos ... 0000'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/runs13/hornet_0000_m_opth_2_60_ce_agabor_0/hornet_0000.pth --version 0 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_0000_m_opth_2_60_ce_agabor_0/hornet_0000_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 0001'
# python3.8 main.py --train --load_model ./runs_models/runs13/hornet_0001_m_opth_2_60_ce_agabor_1/hornet_0001.pth --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1 --dump ./runs_models/runs13/hornet_0001_m_opth_2_60_ce_agabor_1/hornet_0001.pth --version 1 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_0001_m_opth_2_60_ce_agabor_1/hornet_0001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 0011'
# python3.8 main.py --train --load_model ./runs_models/runs13/hornet_0011_m_opth_2_60_ce_agabor_2/hornet_0011.pth --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs13/hornet_0011_m_opth_2_60_ce_agabor_2/hornet_0011.pth --version 2 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_0011_m_opth_2_60_ce_agabor_2/hornet_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 0101'
# python3.8 main.py --train --load_model ./runs_models/runs13/hornet_0101_m_opth_2_60_ce_agabor_3/hornet_0101.pth --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 0 1 --dump ./runs_models/runs13/hornet_0101_m_opth_2_60_ce_agabor_3/hornet_0101.pth --version 3 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_0101_m_opth_2_60_ce_agabor_3/hornet_0101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 0111'
# python3.8 main.py --train --load_model ./runs_models/runs13/hornet_0111_m_opth_2_60_ce_agabor_4/hornet_0111.pth --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 1 1 --dump ./runs_models/runs13/hornet_0111_m_opth_2_60_ce_agabor_4/hornet_0111.pth --version 4 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_0111_m_opth_2_60_ce_agabor_4/hornet_0111_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 1001'
# python3.8 main.py --train --load_model ./runs_models/runs13/hornet_1001_m_opth_2_60_ce_agabor_5/hornet_1001.pth --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 0 1 --dump ./runs_models/runs13/hornet_1001_m_opth_2_60_ce_agabor_5/hornet_1001.pth --version 5 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_1001_m_opth_2_60_ce_agabor_5/hornet_1001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 1011'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 1 1 --dump ./runs_models/runs13/hornet_1011_m_opth_2_60_ce_agabor_6/hornet_1011.pth --version 6 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_1011_m_opth_2_60_ce_agabor_6/hornet_1011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 1101'
# python3.8 main.py --train  --load_model ./runs_models/runs13/hornet_1101_m_opth_2_60_ce_agabor_7/hornet_1101.pth --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs13/hornet_1101_m_opth_2_60_ce_agabor_7/hornet_1101.pth --version 7 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_1101_m_opth_2_60_ce_agabor_7/hornet_1101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# echo 'Entrenamientos ... 1111'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 1 1 --dump ./runs_models/runs13/hornet_1111_m_opth_2_60_ce_agabor_8/hornet_1111.pth --version 8 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_1111_m_opth_2_60_ce_agabor_8/hornet_1111_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# Convnext
# Evaluaciones para obtener los softmax scores

# for i in runs_models/runs14/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_small_0001_best.pth runs_models/runs14/convnext_0011_m_opth_2_60_ce_agabor_2/convnext_small_0011_best.pth runs_models/runs14/convnext_0101_m_opth_2_60_ce_agabor_3/convnext_small_0101_best.pth runs_models/runs14/convnext_0111_m_opth_2_60_ce_agabor_4/convnext_small_0111_best.pth runs_models/runs14/convnext_1001_m_opth_2_60_ce_agabor_5/convnext_small_1001_best.pth runs_models/runs14/convnext_1011_m_opth_2_60_ce_agabor_6/convnext_small_1011_best.pth runs_models/runs14/convnext_1101_m_opth_2_60_ce_agabor_7/convnext_small_1101_best.pth runs_models/runs14/convnext_1111_m_opth_2_60_ce_agabor_8/convnext_small_1111_best.pth; do
#     python3.8 main.py --matrix --set test --load_model $i --msg JSONFiles/DDR_M_C/DDR_test.json --device 0
# done;

# for i in runs_models/runs14/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_small_0001_best.pth runs_models/runs14/convnext_0011_m_opth_2_60_ce_agabor_2/convnext_small_0011_best.pth runs_models/runs14/convnext_0101_m_opth_2_60_ce_agabor_3/convnext_small_0101_best.pth runs_models/runs14/convnext_0111_m_opth_2_60_ce_agabor_4/convnext_small_0111_best.pth runs_models/runs14/convnext_1001_m_opth_2_60_ce_agabor_5/convnext_small_1001_best.pth runs_models/runs14/convnext_1011_m_opth_2_60_ce_agabor_6/convnext_small_1011_best.pth runs_models/runs14/convnext_1101_m_opth_2_60_ce_agabor_7/convnext_small_1101_best.pth runs_models/runs14/convnext_1111_m_opth_2_60_ce_agabor_8/convnext_small_1111_best.pth; do
#     python3.8 main.py --matrix --set valid --load_model $i --msg JSONFiles/DDR_M_C/DDR_valid.json --device 0
# done;

# for i in runs_models/runs14/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_small_0001_best.pth runs_models/runs14/convnext_0011_m_opth_2_60_ce_agabor_2/convnext_small_0011_best.pth runs_models/runs14/convnext_0101_m_opth_2_60_ce_agabor_3/convnext_small_0101_best.pth runs_models/runs14/convnext_0111_m_opth_2_60_ce_agabor_4/convnext_small_0111_best.pth runs_models/runs14/convnext_1001_m_opth_2_60_ce_agabor_5/convnext_small_1001_best.pth runs_models/runs14/convnext_1011_m_opth_2_60_ce_agabor_6/convnext_small_1011_best.pth runs_models/runs14/convnext_1101_m_opth_2_60_ce_agabor_7/convnext_small_1101_best.pth runs_models/runs14/convnext_1111_m_opth_2_60_ce_agabor_8/convnext_small_1111_best.pth; do
#     python3.8 main.py --matrix --set train --load_model $i --msg JSONFiles/DDR_M_C/DDR_train.json --device 0
# done;

# Hornet
# Evaluaciones para obtener los softmax scores

# for i in runs_models/runs13/hornet_0000_m_opth_2_60_ce_agabor_0/hornet_0000_best.pth runs_models/runs13/hornet_0001_m_opth_2_60_ce_agabor_1/hornet_0001_best.pth runs_models/runs13/hornet_0011_m_opth_2_60_ce_agabor_2/hornet_0011_best.pth runs_models/runs13/hornet_0101_m_opth_2_60_ce_agabor_3/hornet_0101_best.pth runs_models/runs13/hornet_0111_m_opth_2_60_ce_agabor_4/hornet_0111_best.pth runs_models/runs13/hornet_1001_m_opth_2_60_ce_agabor_5/hornet_1001_best.pth runs_models/runs13/hornet_1011_m_opth_2_60_ce_agabor_6/hornet_1011_best.pth runs_models/runs13/hornet_1101_m_opth_2_60_ce_agabor_7/hornet_1101_best.pth runs_models/runs13/hornet_1111_m_opth_2_60_ce_agabor_8/hornet_1111_best.pth; do
#     python3.8 main.py --matrix --set test --load_model $i --msg JSONFiles/DDR_M_H/DDR_test.json --device 0
# done;

# for i in runs_models/runs13/hornet_0000_m_opth_2_60_ce_agabor_0/hornet_0000_best.pth runs_models/runs13/hornet_0001_m_opth_2_60_ce_agabor_1/hornet_0001_best.pth runs_models/runs13/hornet_0011_m_opth_2_60_ce_agabor_2/hornet_0011_best.pth runs_models/runs13/hornet_0101_m_opth_2_60_ce_agabor_3/hornet_0101_best.pth runs_models/runs13/hornet_0111_m_opth_2_60_ce_agabor_4/hornet_0111_best.pth runs_models/runs13/hornet_1001_m_opth_2_60_ce_agabor_5/hornet_1001_best.pth runs_models/runs13/hornet_1011_m_opth_2_60_ce_agabor_6/hornet_1011_best.pth runs_models/runs13/hornet_1101_m_opth_2_60_ce_agabor_7/hornet_1101_best.pth runs_models/runs13/hornet_1111_m_opth_2_60_ce_agabor_8/hornet_1111_best.pth; do
#     python3.8 main.py --matrix --set valid --load_model $i --msg JSONFiles/DDR_M_H/DDR_valid.json --device 0
# done;

# for i in runs_models/runs13/hornet_0000_m_opth_2_60_ce_agabor_0/hornet_0000_best.pth runs_models/runs13/hornet_0001_m_opth_2_60_ce_agabor_1/hornet_0001_best.pth runs_models/runs13/hornet_0011_m_opth_2_60_ce_agabor_2/hornet_0011_best.pth runs_models/runs13/hornet_0101_m_opth_2_60_ce_agabor_3/hornet_0101_best.pth runs_models/runs13/hornet_0111_m_opth_2_60_ce_agabor_4/hornet_0111_best.pth runs_models/runs13/hornet_1001_m_opth_2_60_ce_agabor_5/hornet_1001_best.pth runs_models/runs13/hornet_1011_m_opth_2_60_ce_agabor_6/hornet_1011_best.pth runs_models/runs13/hornet_1101_m_opth_2_60_ce_agabor_7/hornet_1101_best.pth runs_models/runs13/hornet_1111_m_opth_2_60_ce_agabor_8/hornet_1111_best.pth; do
#     python3.8 main.py --matrix --set train --load_model $i --msg JSONFiles/DDR_M_H/DDR_train.json --device 0
# done;

# python3.8 main.py --dataloader_json ./JSONFiles/DDR_M_H --snf_eval --load_model ./runs_models/runSNF5/SNF_12_C.pthbest.pth --device 1

# python3.8 main.py --eval --load_model runs_models/runs14/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_small_0001_best.pth --dataloader_json JSONFiles/messidor2/messidor2_ --set test --att --device 1


#echo 'Entrenamientos internimage ... 0011 con eyepacs'
#python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs15/internimage_0011_m_opth_2_60_ce_agabor_1/internimage_0011.pth --version 1 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
#python3.8 main.py --eval --load_model ./runs_models/runs15/internimage_0011_m_opth_2_60_ce_agabor_1/internimage_0011_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0

# echo 'Entrenamientos hornet ... 0001 con eyepacs'
# python3.8 main.py --train --epochs 60 --load_model ./runs_models/runs15/hornet_0001_m_opth_2_60_ce_agabor_1/hornet_0001.pth --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1 --dump ./runs_models/runs15/hornet_0001_m_opth_2_60_ce_agabor_1/hornet_0001.pth --version 1 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs15/hornet_0001_m_opth_2_60_ce_agabor_1/hornet_0001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
 
# echo 'Entrenamientos hornet ... 0000 con eyepacs'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.9 --patience 5 --att --attn_block 0 0 0 0 --dump ./runs_models/runs15/hornet_0000_m_opth_2_60_ce_agabor_0/hornet_0000.pth --version 0 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs15/hornet_0000_m_opth_2_60_ce_agabor_0/hornet_0000_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0

# echo 'Entrenamientos ... 0000 ... InterImage'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/runs16/internimage_0000_ce_0/internimage_0011.pth --version 0
# python3.8 main.py --eval --load_model ./runs_models/runs16/internimage_0000_ce_0/internimage_0000_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# # Flag 6

# echo 'Entrenamientos ... 0001 ... InterImage este es con gabor'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1 --dump ./runs_models/runs16/internimage_0001_ce_3/internimage_0001.pth --version 3
# python3.8 main.py --eval --load_model ./runs_models/runs16/internimage_0001_ce_3/internimage_0001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
 
# # Flag 4

# echo 'Entrenamientos ... 0001 ... InterImage + CE + Mopht2'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1 --dump ./runs_models/runs16/internimage_0001_ce_mo2_4/internimage_0001.pth --version 4 --loss_sensitive --loss_mode 3 --base_loss ce
# python3.8 main.py --eval --load_model ./runs_models/runs16/internimage_0001_ce_mo2_4/internimage_0001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# # Flag 5

# echo 'Entrenamientos ... 0011 ... InterImage + CE + Mopht2'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs16/internimage_0011_ce_mo2_5/internimage_0011.pth --version 5 --loss_sensitive --loss_mode 3 --base_loss ce
# python3.8 main.py --eval --load_model ./runs_models/runs16/internimage_0011_ce_mo2_5/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# echo 'Entrenamientos ... 0001 ... Gabor - InterImage + CE'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1 --dump ./runs_models/runs16/internimage_0011_ce_7/internimage_0011.pth --version 6 --gabor 2 --no_g 1
# python3.8 main.py --eval --load_model ./runs_models/runs16/internimage_0011_ce_7/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# Flag 7

# echo 'Entrenamientos ... 0001 ... Gabor - InterImage + CE + Mopht2'
# python3.8 main.py --train --load_model runs_models/runs16/internimage_0001_ce_mo2_6/internimage_0001.pth --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1 --dump ./runs_models/runs16/internimage_0001_ce_mo2_6/internimage_0001.pth --version 6 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1
# python3.8 main.py --eval --load_model ./runs_models/runs16/internimage_0001_ce_mo2_6/internimage_0001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
 
# # Flag 8

# echo 'Entrenamientos ... 0011 ... InterImage + Gabor + CE'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs16/internimage_0011_ce_8/internimage_0011.pth --version 8 --gabor 2 --no_g 1
# python3.8 main.py --eval --load_model ./runs_models/runs16/internimage_0011_ce_8/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# # Flag 3

# echo 'Entrenamientos ... 0011 ... InterImage + CE'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs16/internimage_0011_ce_9/internimage_0011.pth --version 9
# python3.8 main.py --eval --load_model ./runs_models/runs16/internimage_0011_ce_9/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# echo 'Entrenamientos ... 0000 ... InterImage + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/runs21/internimage_0000_mopth2_0/internimage_0000.pth --version 0 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0000_mopth2_0/internimage_0000_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0000_mopth2_0/internimage_0000_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0000_mopth2_0/internimage_0000_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 0001 ... InterImage + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1  --dump ./runs_models/runs21/internimage_0001_mopth2_1/internimage_0001.pth --version 1 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0001_mopth2_1/internimage_0001_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0001_mopth2_1/internimage_0001_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0001_mopth2_1/internimage_0001_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 0011 ... InterImage + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs21/internimage_0011_mopth2_2/internimage_0011.pth --version 2 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0011_mopth2_2/internimage_0011_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0011_mopth2_2/internimage_0011_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0011_mopth2_2/internimage_0011_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# # 
# echo 'Entrenamientos ... 0101 ... InterImage + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 0 1 --dump ./runs_models/runs21/internimage_0101_mopth2_3/internimage_0101.pth --version 3 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0101_mopth2_3/internimage_0101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0101_mopth2_3/internimage_0101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0101_mopth2_3/internimage_0101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# echo 'Entrenamientos ... 0111 ... InterImage + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 1 1 --dump ./runs_models/runs21/internimage_0111_mopth2_4/internimage_0111.pth --version 4 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0111_mopth2_4/internimage_0111_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0111_mopth2_4/internimage_0111_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_0111_mopth2_4/internimage_0111_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# # echo 'Entrenamientos ... 1001 ... InterImage + Mopth2'

# # python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 0 1 --dump ./runs_models/runs17/internimage_1001_mopth2_5/internimage_1001.pth --version 5 --warm_up 15 --save_per_epoch 20 30 40
# # # Acc
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1001_mopth2_5/internimage_1001_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # # AA
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1001_mopth2_5/internimage_1001_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # # WK
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1001_mopth2_5/internimage_1001_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# # echo 'Entrenamientos ... 1011 ... InterImage + Mopth2'

# # python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 1 1 --dump ./runs_models/runs17/internimage_1011_mopth2_6/internimage_1011.pth --version 6 --warm_up 15 --save_per_epoch 20 30 40
# # # Acc
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1011_mopth2_6/internimage_1011_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # # AA
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1011_mopth2_6/internimage_1011_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # # WK
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1011_mopth2_6/internimage_1011_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# # echo 'Entrenamientos ... 1101 ... InterImage + Mopth2'

# # python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs17/internimage_1101_mopth2_7/internimage_1011.pth --version 7 --warm_up 15 --save_per_epoch 20 30 40
# # # Acc
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1101_mopth2_7/internimage_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # # AA
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1101_mopth2_7/internimage_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # # WK
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1101_mopth2_7/internimage_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# # echo 'Entrenamientos ... 1111 ... InterImage + Mopth2'

# # python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 1 1 --dump ./runs_models/runs17/internimage_1111_mopth2_8/internimage_1111.pth --version 8 --warm_up 15 --save_per_epoch 20 30 40
# # # Acc
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1111_mopth2_8/internimage_1111_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # # AA
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1111_mopth2_8/internimage_1111_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # # WK
# # python3.8 main.py --eval --load_model ./runs_models/runs17/internimage_1111_mopth2_8/internimage_1111_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1011 ... convnext_small_ + Mopth2'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00002 --decay_lr 0.3 --patience 3 --att --attn_block 1 0 1 1 --dump ./runs_models/runs18/convnext_small_1011_mopth2_6/convnext_small_1011.pth --version 6 --warm_up 5 
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs18/convnext_small_1011_mopth2_6/convnext_small_1011_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs18/convnext_small_1011_mopth2_6/convnext_small_1011_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs18/convnext_small_1011_mopth2_6/convnext_small_1011_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1101 ... convnext_small_ + Mopth2'

# python3.8 main.py --train --epochs 30 --batch --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00002 --decay_lr 0.3 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs18/convnext_small_1101_mopth2_7/convnext_small_1011.pth --version 7 --warm_up 5
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs18/convnext_small_1101_mopth2_7/convnext_small_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs18/convnext_small_1101_mopth2_7/convnext_small_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs18/convnext_small_1101_mopth2_7/convnext_small_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# echo 'Entrenamientos ... 1111 ... convnext_small_ + Mopth2'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00002 --decay_lr 0.3 --patience 3 --att --attn_block 1 1 1 1 --dump ./runs_models/runs18/convnext_small_1111_mopth2_8/convnext_small_1111.pth --version 8 --warm_up 5 
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs18/convnext_small_1111_mopth2_8/convnext_small_1111_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs18/convnext_small_1111_mopth2_8/convnext_small_1111_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs18/convnext_small_1111_mopth2_8/convnext_small_1111_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0




# echo 'Entrenamientos ... 0000 ... hornet_ + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/runs19/hornet_0000_mopth2_0/hornet_0000.pth --version 0 --warm_up 15 --save_per_epoch 20 30 40
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0000_mopth2_0/hornet_0000_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0000_mopth2_0/hornet_0000_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0000_mopth2_0/hornet_0000_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 0001 ... hornet_ + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1 --dump ./runs_models/runs19/hornet_0001_mopth2_1/hornet_0001.pth --version 1 --warm_up 15 --save_per_epoch 20 30 40
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0001_mopth2_1/hornet_0001_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0001_mopth2_1/hornet_0001_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0001_mopth2_1/hornet_0001_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 0011 ... hornet_ + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs19/hornet_0011_mopth2_2/hornet_0011.pth --version 2 --warm_up 15 --save_per_epoch 20 30 40
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0011_mopth2_2/hornet_0011_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0011_mopth2_2/hornet_0011_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0011_mopth2_2/hornet_0011_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 0101 ... hornet_ + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 0 1 --dump ./runs_models/runs19/hornet_0101_mopth2_3/hornet_0101.pth --version 3 --warm_up 15 --save_per_epoch 20 30 40
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0101_mopth2_3/hornet_0101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0101_mopth2_3/hornet_0101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0101_mopth2_3/hornet_0101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# echo 'Entrenamientos ... 0111 ... hornet_ + Mopth2'

# python3.8 main.py --train --load_model ./runs_models/runs19/hornet_0111_mopth2_4/hornet_0111.pth --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 1 1 --dump ./runs_models/runs19/hornet_0111_mopth2_4/hornet_0111.pth --version 4 --warm_up 15 --save_per_epoch 20 30 40
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0111_mopth2_4/hornet_0111_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0111_mopth2_4/hornet_0111_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_0111_mopth2_4/hornet_0111_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# echo 'Entrenamientos ... 1001 ... hornet_ + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 0 1 --dump ./runs_models/runs19/hornet_1001_mopth2_5/hornet_1001.pth --load_model ./runs_models/runs19/hornet_1001_mopth2_5/hornet_1001.pth --version 5 --warm_up 15 --save_per_epoch 20 30 40
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_1001_mopth2_5/hornet_1001_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_1001_mopth2_5/hornet_1001_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_1001_mopth2_5/hornet_1001_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1011 ... hornet_ + Mopth2'

# python3.8 main.py --train --epochs 60 --load_model ./runs_models/runs19/hornet_1011_mopth2_6/hornet_1011.pth --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 1 1 --dump ./runs_models/runs19/hornet_1011_mopth2_6/hornet_1011.pth --version 6 --warm_up 15 --save_per_epoch 20 30 40
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_1011_mopth2_6/hornet_1011_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_1011_mopth2_6/hornet_1011_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs19/hornet_1011_mopth2_6/hornet_1011_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

#!/bin/bash

# InterImage

#Acc
# contador=0
# for i in 0000; do
#      python3.8 main.py --matrix --set train --load_model runs_models/runs17/internimage_"$i"_mopth2_"$contador"/internimage_"$i"_best_acc.pth --msg JSONFiles/finales/ddr_intern_acc/DDR_train.json --device 0 --tipo_head att_i
#      let contador++
# done;

# contador1=0
# for i in 0000; do
#      python3.8 main.py --matrix --set valid --load_model runs_models/runs17/internimage_"$i"_mopth2_"$contador1"/internimage_"$i"_best_acc.pth --msg JSONFiles/finales/ddr_intern_acc/DDR_valid.json --device 0 --tipo_head att_i
#      let contador1++
# done;

# contador2=0
# for i in 0000; do
#      python3.8 main.py --matrix --set test --load_model runs_models/runs17/internimage_"$i"_mopth2_"$contador2"/internimage_"$i"_best_acc.pth --msg JSONFiles/finales/ddr_intern_acc/DDR_test.json --device 0 --tipo_head att_i
#      let contador2++
# done;

# # AA

# contador3=0
# for i in 0000; do
#      python3.8 main.py --matrix --set train --load_model runs_models/runs17/internimage_"$i"_mopth2_"$contador3"/internimage_"$i"_best_aa.pth --msg JSONFiles/finales/ddr_intern_aa/DDR_train.json --device 0 --tipo_head att_i
#      let contador3++
# done;

# contador4=0
# for i in 0000; do
#      python3.8 main.py --matrix --set valid --load_model runs_models/runs17/internimage_"$i"_mopth2_"$contador4"/internimage_"$i"_best_aa.pth --msg JSONFiles/finales/ddr_intern_aa/DDR_valid.json --device 0  --tipo_head att_i
#      let contador4++
# done;

# contador5=0
# for i in 0000; do
#      python3.8 main.py --matrix --set test --load_model runs_models/runs17/internimage_"$i"_mopth2_"$contador5"/internimage_"$i"_best_aa.pth --msg JSONFiles/finales/ddr_intern_aa/DDR_test.json --device 0 --tipo_head att_i
#      let contador5++
# done;

# # WK

# contador6=0
# for i in  0000; do
#      python3.8 main.py --matrix --set train --load_model runs_models/runs17/internimage_"$i"_mopth2_"$contador6"/internimage_"$i"_best_wk.pth --msg JSONFiles/finales/ddr_intern_wk/DDR_train.json --device 0 --tipo_head att_i
#      let contador6++
# done;

# contador7=0
# for i in 0000; do
#      python3.8 main.py --matrix --set valid --load_model runs_models/runs17/internimage_"$i"_mopth2_"$contador7"/internimage_"$i"_best_wk.pth --msg JSONFiles/finales/ddr_intern_wk/DDR_valid.json --device 0 --tipo_head att_i
#      let contador7++
# done;

# contador8=0
# for i in 0000; do
#      python3.8 main.py --matrix --set test --load_model runs_models/runs17/internimage_"$i"_mopth2_"$contador8"/internimage_"$i"_best_wk.pth --msg JSONFiles/finales/ddr_intern_wk/DDR_test.json --device 0 --tipo_head att_i
#      let contador8++
# done;

# Hornet

#Acc
# contador=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set train --load_model runs_models/runs19/hornet_"$i"_mopth2_"$contador"/hornet_"$i"_best_acc.pth --msg JSONFiles/finales/ddr_hornet_acc/DDR_train.json --device 0 --tipo_head att_h
#      let contador++
# done;

# contador1=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set valid --load_model runs_models/runs19/hornet_"$i"_mopth2_"$contador1"/hornet_"$i"_best_acc.pth --msg JSONFiles/finales/ddr_hornet_acc/DDR_valid.json --device 0 --tipo_head att_h
#      let contador1++
# done;

# contador2=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set test --load_model runs_models/runs19/hornet_"$i"_mopth2_"$contador2"/hornet_"$i"_best_acc.pth --msg JSONFiles/finales/ddr_hornet_acc/DDR_test.json --device 0 --tipo_head att_h
#      let contador2++
# done;

# # AA

# contador3=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set train --load_model runs_models/runs19/hornet_"$i"_mopth2_"$contador3"/hornet_"$i"_best_aa.pth --msg JSONFiles/finales/ddr_hornet_aa/DDR_train.json --device 0 --tipo_head att_h
#      let contador3++
# done;

# contador4=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set valid --load_model runs_models/runs19/hornet_"$i"_mopth2_"$contador4"/hornet_"$i"_best_aa.pth --msg JSONFiles/finales/ddr_hornet_aa/DDR_valid.json --device 0  --tipo_head att_h
#      let contador4++
# done;

# contador5=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set test --load_model runs_models/runs19/hornet_"$i"_mopth2_"$contador5"/hornet_"$i"_best_aa.pth --msg JSONFiles/finales/ddr_hornet_aa/DDR_test.json --device 0 --tipo_head att_h
#      let contador5++
# done;

# # WK

# contador6=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set train --load_model runs_models/runs19/hornet_"$i"_mopth2_"$contador6"/hornet_"$i"_best_wk.pth --msg JSONFiles/finales/ddr_hornet_wk/DDR_train.json --device 0 --tipo_head att_h
#      let contador6++
# done;

# contador7=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set valid --load_model runs_models/runs19/hornet_"$i"_mopth2_"$contador7"/hornet_"$i"_best_wk.pth --msg JSONFiles/finales/ddr_hornet_wk/DDR_valid.json --device 0 --tipo_head att_h
#      let contador7++
# done;

# contador8=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set test --load_model runs_models/runs19/hornet_"$i"_mopth2_"$contador8"/hornet_"$i"_best_wk.pth --msg JSONFiles/finales/ddr_hornet_wk/DDR_test.json --device 0 --tipo_head att_h
#      let contador8++
# done;

# #python3.8 main.py --matrix --set train --load_model runs_models/runs19/hornet_0000_mopth2_0/hornet_0000_best_aa.pth --msg JSONFiles/finales/ddr_hornet_aa/DDR_train.json --device 0 --tipo_head att_h

# echo 'Entrenamientos ... 0011 ... InterImage + Mopth2 eyepacs'

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs20/internimage_0011_mopth2_2/internimage_0011.pth --version 2 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs20/internimage_0011_mopth2_2/internimage_0011_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs20/internimage_0011_mopth2_2/internimage_0011_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs20/internimage_0011_mopth2_2/internimage_0011_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1001 ... InterImage + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 0 1 --load_model ./runs_models/runs21/internimage_1001_mopth2_5/internimage_1001.pth --dump ./runs_models/runs21/internimage_1001_mopth2_5/internimage_1001.pth --version 5 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1001_mopth2_5/internimage_1001_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1001_mopth2_5/internimage_1001_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1001_mopth2_5/internimage_1001_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1011 ... InterImage + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 1 1 --dump ./runs_models/runs21/internimage_1011_mopth2_6/internimage_1011.pth --load_model ./runs_models/runs21/internimage_1011_mopth2_6/internimage_1011.pth --version 6 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1011_mopth2_6/internimage_1011_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1011_mopth2_6/internimage_1011_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1011_mopth2_6/internimage_1011_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1101 ... InterImage + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs21/internimage_1101_mopth2_7/internimage_1011.pth --version 7 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1101_mopth2_7/internimage_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1101_mopth2_7/internimage_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1101_mopth2_7/internimage_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# echo 'Entrenamientos ... 1111 ... InterImage + Mopth2'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 1 1 --dump ./runs_models/runs21/internimage_1111_mopth2_8/internimage_1111.pth --version 8 --warm_up 15 --save_per_epoch 20 30 40 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1111_mopth2_8/internimage_1111_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1111_mopth2_8/internimage_1111_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs21/internimage_1111_mopth2_8/internimage_1111_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# Evaluaciones de los modelos SNF

# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_5.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_7.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_8.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_9.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_10.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_11.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_12.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_13.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_14.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_16.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_17.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_18.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_19.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_aa/DDR_ --load_model runs_models/snf_intern/aa/SNF_19_2.pthbest.pth --device 1

# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_5.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_7.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_8.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_9.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_10.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_11.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_12.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_13.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_14.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_16.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_17.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_18.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_19.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --load_model runs_models/snf_intern/acc/SNF_19_2.pthbest.pth --device 1

#python3.8 main.py --snf_eval --dataloader_json JSONFiles/DDR_M/DDR_ --load_model runs_models/snf_convnext/snfw2.pt --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_7.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_8.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_9.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_10.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_11.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_12.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_13.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_14.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_16.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_17.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_18.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_19.pthbest.pth --device 1
# python3.8 main.py --snf_eval --dataloader_json JSONFiles/finales/ddr_convnext_wk/DDR_ --load_model runs_models/snf_convnext/wk/SNF_19_2.pthbest.pth --device 1


# echo 'Entrenamientos ... 0001 ... convnext_small_ + Mopth2 -- 0'

# python3.8 main.py --train --epochs 30  --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00002 --decay_lr 0.3 --patience 3 --att --attn_block 0 0 0 1 --dump ./runs_models/runs40/convnext_small_0001_mopth2_0/convnext_small_0001.pth --version 0 --warm_up 5 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0001_mopth2_0/convnext_small_0001_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0001_mopth2_0/convnext_small_0001_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0001_mopth2_0/convnext_small_0001_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# #Pruebas con ConvNeXt 6 casos

# echo 'Entrenamientos ... 1101 ... convnext_small_ + Mopth2  -- 1'

# python3.8 main.py --train --epochs 30  --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00002 --decay_lr 0.3 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs40/convnext_small_1101_mopth2_1/convnext_small_1011.pth --version 1 --warm_up 5 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_1/convnext_small_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_1/convnext_small_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_1/convnext_small_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0




# echo 'Entrenamientos ... 0000 ... convnext_small_ + Mopth2 --- 2 - 3'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00002 --decay_lr 0.3 --patience 3 --att --attn_block 0 0 0 0 --dump ./runs_models/runs40/convnext_small_0000_mopth2_23/convnext_small_0000.pth --version 23 --warm_up 5 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0000_mopth2_23/convnext_small_0000_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0000_mopth2_23/convnext_small_0000_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0000_mopth2_23/convnext_small_0000_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1101 ... convnext_small_ + Mopth2  -- 0'

# python3.8 main.py --train --load_model ./runs_models/runs40/convnext_small_1101_mopth2_0/convnext_small_1011.pth --epochs 70  --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000001 --decay_lr 0.7 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs40/convnext_small_1101_mopth2_0/convnext_small_1011.pth --version 0 --warm_up 12 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_0/convnext_small_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_0/convnext_small_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_0/convnext_small_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

# echo 'Entrenamientos ... 1101 ... convnext_small_ + Mopth2  -- 1'

# python3.8 main.py --train --load_model ./runs_models/runs40/convnext_small_1101_mopth2_1/convnext_small_1011.pth --epochs 70  --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000002 --decay_lr 0.5 --patience 5 --att --attn_block 1 1 0 1 --dump ./runs_models/runs40/convnext_small_1101_mopth2_1/convnext_small_1011.pth --version 1 --warm_up 12 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_1/convnext_small_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_1/convnext_small_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_1/convnext_small_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1101 ... convnext_small_ + Mopth2  -- 2'

# python3.8 main.py --train --epochs 30  --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.6 --patience 4 --att --attn_block 1 1 0 1 --dump ./runs_models/runs40/convnext_small_1101_mopth2_2/convnext_small_1011.pth --version 2 --warm_up 15 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_8/convnext_small_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_8/convnext_small_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_8/convnext_small_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1101 ... convnext_small_ + Mopth2  -  3'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00001 --decay_lr 0.5 --patience 4 --att --attn_block 1 1 0 1 --dump ./runs_models/runs40/convnext_small_1101_mopth2_14/convnext_small_1011.pth --version 14 --warm_up 8 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_14/convnext_small_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_14/convnext_small_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_14/convnext_small_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1101 ... convnext_small_ + Mopth2  -  3'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00001 --decay_lr 0.9 --patience 4 --att --attn_block 1 1 0 1 --dump ./runs_models/runs40/convnext_small_1101_mopth2_15/convnext_small_1011.pth --version 15 --warm_up 5 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_15/convnext_small_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_15/convnext_small_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_15/convnext_small_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 1101 ... convnext_small_ + Mopth2  -  3'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00001 --decay_lr 0.2 --patience 4 --att --attn_block 1 1 0 1 --dump ./runs_models/runs40/convnext_small_1101_mopth2_16/convnext_small_1011.pth --version 16 --warm_up 20 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_16/convnext_small_1101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_16/convnext_small_1101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_1101_mopth2_16/convnext_small_1101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 0000 ... convnext_small_ + Mopth2  -  0'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00001 --decay_lr 0.6 --patience 4 --att --attn_block 0 0 0 0 --dump ./runs_models/runs40/convnext_small_0000_mopth2_0/convnext_small_0000.pth --version 0 --warm_up 10 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0000_mopth2_0/convnext_small_0000_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0000_mopth2_0/convnext_small_0000_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0000_mopth2_0/convnext_small_0000_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 0001 ... convnext_small_ + Mopth2  -  0'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00001 --decay_lr 0.6 --patience 4 --att --attn_block 0 0 0 1 --dump ./runs_models/runs40/convnext_small_0001_mopth2_1/convnext_small_0001.pth --version 1 --warm_up 10 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0001_mopth2_1/convnext_small_0001_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0001_mopth2_1/convnext_small_0001_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0001_mopth2_1/convnext_small_0001_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 0011 ... convnext_small_ + Mopth2  -  2'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00001 --decay_lr 0.6 --patience 4 --att --attn_block 0 0 1 1 --dump ./runs_models/runs40/convnext_small_0011_mopth2_2/convnext_small_0011.pth --version 2 --warm_up 10 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0011_mopth2_2/convnext_small_0011_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0011_mopth2_2/convnext_small_0011_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0011_mopth2_2/convnext_small_0011_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# echo 'Entrenamientos ... 0101 ... convnext_small_ + Mopth2  -  3'

# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 0  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.00001 --decay_lr 0.6 --patience 4 --att --attn_block 0 1 0 1 --dump ./runs_models/runs40/convnext_small_0101_mopth2_3/convnext_small_0101.pth --version 3 --warm_up 10 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0101_mopth2_3/convnext_small_0101_best_acc.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0101_mopth2_3/convnext_small_0101_best_aa.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0101_mopth2_3/convnext_small_0101_best_wk.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0


# ConvNeXt

# #Acc
# contador=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set train --load_model runs_models/runs40/convnext_small_"$i"_mopth2_"$contador"/convnext_small_"$i"_best_acc.pth --msg JSONFiles/finales_convnext/ddr_convnext_acc/DDR_train.json --device 1 --tipo_head att_c
#      let contador++
# done;

# contador1=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set valid --load_model runs_models/runs40/convnext_small_"$i"_mopth2_"$contador1"/convnext_small_"$i"_best_acc.pth --msg JSONFiles/finales_convnext/ddr_convnext_acc/DDR_valid.json --device 1 --tipo_head att_c
#      let contador1++
# done;

# contador2=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set test --load_model runs_models/runs40/convnext_small_"$i"_mopth2_"$contador2"/convnext_small_"$i"_best_acc.pth --msg JSONFiles/finales_convnext/ddr_convnext_acc/DDR_test.json --device 1 --tipo_head att_c
#      let contador2++
# done;

# # AA

# contador3=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set train --load_model runs_models/runs40/convnext_small_"$i"_mopth2_"$contador3"/convnext_small_"$i"_best_aa.pth --msg JSONFiles/finales_convnext/ddr_convnext_aa/DDR_train.json --device 1 --tipo_head att_c
#      let contador3++
# done;

# contador4=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set valid --load_model runs_models/runs40/convnext_small_"$i"_mopth2_"$contador4"/convnext_small_"$i"_best_aa.pth --msg JSONFiles/finales_convnext/ddr_convnext_aa/DDR_valid.json --device 1  --tipo_head att_c
#      let contador4++
# done;

# contador5=0
# for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
#      python3.8 main.py --matrix --set test --load_model runs_models/runs40/convnext_small_"$i"_mopth2_"$contador5"/convnext_small_"$i"_best_aa.pth --msg JSONFiles/finales_convnext/ddr_convnext_aa/DDR_test.json --device 1 --tipo_head att_c
#      let contador5++
# done;


# # ConvNeXt

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.2 --decay_lr 0.6 --patience 100 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_12_2.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_19_2.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 0.3 --decay_lr 0.9 --patience 300 --epochs 800 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_5.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.3 --decay_lr 0.9 --patience 200 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_7.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 2.2 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_8.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 2.1 --decay_lr 0.8 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_9.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 0.9 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_10.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 0.9 --decay_lr 0.7 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_11.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.5 --decay_lr 0.6 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_12.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.5 --decay_lr 0.6 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_13.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.2 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_14.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.2 --decay_lr 0.7 --patience 90 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_15.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_16.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_17.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.1 --decay_lr 0.9 --patience 80 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_18.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_19.pth --loss_sensitive --loss_mode 3


# acc

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.2 --decay_lr 0.6 --patience 100 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_12_2.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_19_2.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 0.3 --decay_lr 0.9 --patience 300 --epochs 800 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_5.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.3 --decay_lr 0.9 --patience 200 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_7.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 2.2 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_8.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 2.1 --decay_lr 0.8 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_9.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 0.9 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_10.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 0.9 --decay_lr 0.7 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_11.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.5 --decay_lr 0.6 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_12.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.5 --decay_lr 0.6 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_13.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.2 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_14.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.2 --decay_lr 0.7 --patience 90 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_15.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_16.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_17.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.1 --decay_lr 0.9 --patience 80 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_18.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/acc/SNF_19.pth --loss_sensitive --loss_mode 3


# # wk

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.2 --decay_lr 0.6 --patience 100 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_12_2.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_19_2.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 0.3 --decay_lr 0.9 --patience 300 --epochs 800 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_5.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.3 --decay_lr 0.9 --patience 200 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_7.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 2.2 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_8.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 2.1 --decay_lr 0.8 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_9.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 0.9 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_10.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 0.9 --decay_lr 0.7 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_11.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.5 --decay_lr 0.6 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_12.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.5 --decay_lr 0.6 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_13.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.2 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_14.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.2 --decay_lr 0.7 --patience 90 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_15.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_16.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_17.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.1 --decay_lr 0.9 --patience 80 --epochs 400 --batch 128 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_18.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/snf_convnext_f/wk/SNF_19.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --snf_train_custom --lr 0.5 --decay_lr 0.8            --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/snf_convnext_f/aa/SNF_12_2.pth --loss_sensitive --loss_mode 3 --custom  1 2 3 4 6 7 8 

# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0011_mopth2_2/convnext_small_0011.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0011_mopth2_2/convnext_small_0011.pth --dataloader_json JSONFiles/DDR/DDR_  --set valid --att --device 0


# # Acc
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0101_mopth2_3/convnext_small_0101.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/runs40/convnext_small_0101_mopth2_3/convnext_small_0101.pth --dataloader_json JSONFiles/DDR/DDR_  --set valid --att --device 0


# #
# # 1
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9  --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8  --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.2 --decay_lr 0.8  --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8  --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.7--decay_lr 0.8  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2  --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/1/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  1 10 18
# # 2
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9  --patience 40  --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8  --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.2 --decay_lr 0.8  --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8  --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.7--decay_lr 0.8  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2  --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/2/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 16 13 7 23
# # 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9   --patience 40  --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8    --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union ---lr 0.2 --decay_lr 0.8   --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8    --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.7--decay_lr 0.8     --patience 30 --epochs 200 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2      --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/3/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  25 11 23 1 0
# # 6
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9   --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8    --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.2 --decay_lr 0.8   --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8    --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.7--decay_lr 0.8     --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2      --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/6/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 26
# 7
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9   --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8    --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.2 --decay_lr 0.8   --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8    --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.7--decay_lr 0.8     --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2      --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/7/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  2 24 26 23 21 3
# # 8
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9   --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8    --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.2 --decay_lr 0.8   --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8    --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.7--decay_lr 0.8     --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2      --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/8/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12 26
# # 10
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9   --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8    --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.2 --decay_lr 0.8   --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8    --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.7--decay_lr 0.8     --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2      --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/10/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  0 25 16 13 26 1 23 20 12

# # 4
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9   --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8    --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.2 --decay_lr 0.8   --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8    --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.7 --decay_lr 0.8     --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2      --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/4/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  0 1 25 7 23
# # 5
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9   --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8    --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.2 --decay_lr 0.8   --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8    --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.7 --decay_lr 0.8     --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12
# python3.8 main.py --dataloader_json JSONFiles/finales_convnext/ddr_convnext_acc/DDR_ --dataloader_json_2 JSONFiles/finales_convnext/ddr_convnext_aa/DDR_ --dataloader_json_3 JSONFiles/finales_convnext/ddr_convnext_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2      --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/5/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  13 17 0 25 12

# # 3
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9   --patience 40  --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_2.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8    --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_3.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union ---lr 0.2 --decay_lr 0.8   --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_4.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8    --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_5.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union --lr 0.7--decay_lr 0.8     --patience 30 --epochs 200 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_6.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_7.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_8.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_9.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_hornet_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_hornet_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_hornet_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2      --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/h/criterio_10.pth --loss_sensitive --loss_mode 3 --custom  8 26 23 20 18


# # 3
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union --lr 0.015 --decay_lr 0.9  --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_1.pth --loss_sensitive --loss_mode 3 --custom  24 3 21 18 0
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union --lr 0.05 --decay_lr 0.9   --patience 40  --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_2.pth --loss_sensitive --loss_mode 3 --custom   24 3 21 18 0
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union --lr 0.1 --decay_lr 0.8    --patience 40 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_3.pth --loss_sensitive --loss_mode 3 --custom   24 3 21 18 0
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union ---lr 0.2 --decay_lr 0.8   --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_4.pth --loss_sensitive --loss_mode 3 --custom   24 3 21 18 0
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union --lr 0.5 --decay_lr 0.8    --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_5.pth --loss_sensitive --loss_mode 3 --custom   24 3 21 18 0
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union --lr 0.7--decay_lr 0.8     --patience 30 --epochs 200 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_6.pth --loss_sensitive --loss_mode 3 --custom   24 3 21 18 0
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union --lr 0.8 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_7.pth --loss_sensitive --loss_mode 3 --custom   24 3 21 18 0
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union --lr 0.9 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_8.pth --loss_sensitive --loss_mode 3 --custom   24 3 21 18 0
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union --lr 1.2 --decay_lr 0.8    --patience 30 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_9.pth --loss_sensitive --loss_mode 3 --custom   24 3 21 18 0
# python3.8 main.py --dataloader_json JSONFiles/finales/ddr_intern_acc/DDR_ --dataloader_json_2 JSONFiles/finales/ddr_intern_aa/DDR_ --dataloader_json_3 JSONFiles/finales/ddr_intern_wk/DDR_ --snf_train_custom_union --lr 1 --decay_lr 0.2      --patience 50 --epochs 200 --batch 512 --device 1 --dump ./runs_models/runs_union/i/criterio_10.pth --loss_sensitive --loss_mode 3 --custom   24 3 21 18 0

# Entrenamientos con CE + Mopth2 InternImage EyePacs

# echo 'Entrenamientos ... 1101 ... InterImage + Mopth2 ... 0'

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/last_dance/internimage_1101_0/internimage_1011.pth --version 0 --warm_up 10 --save_per_epoch 30 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_1101_0/internimage_1101_best_acc.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_1101_0/internimage_1101_best_aa.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_1101_0/internimage_1101_best_wk.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0

# echo 'Entrenamientos ... 1101 ... InterImage + Mopth2 ... 1'

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 1 1 --dump ./runs_models/last_dance/internimage_1011_1/internimage_1011.pth --version 1 --warm_up 10 --save_per_epoch 30 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_1011_1/internimage_1011_best_acc.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_1011_1/internimage_1011_best_aa.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_1011_1/internimage_1011_best_wk.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0

# echo 'Entrenamientos ... 0000 ... InterImage + Mopth2 ... 2'

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/last_dance/internimage_0000_2/internimage_0000.pth --version 2 --warm_up 10 --save_per_epoch 30 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_0000_2/internimage_0000_best_acc.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_0000_2/internimage_0000_best_aa.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_0000_2/internimage_0000_best_wk.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0

# echo 'Entrenamientos ... 0011 ... InterImage + Mopth2 ... 3'

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/last_dance/internimage_0011_3/internimage_0011.pth --load_model ./runs_models/last_dance/internimage_0011_3/internimage_0011.pth --version 3 --warm_up 10 --save_per_epoch 30 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_0011_3/internimage_0011_best_acc.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_0011_3/internimage_0011_best_aa.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_0011_3/internimage_0011_best_wk.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0

# echo 'Entrenamientos ... 0011 ... InterImage + Mopth2 ... 4'

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 0 1 --dump ./runs_models/last_dance/internimage_0101_4/internimage_0101.pth --version 3 --warm_up 10 --save_per_epoch 30 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_0101_4/internimage_0101_best_acc.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_0101_4/internimage_0101_best_aa.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/last_dance/internimage_0101_4/internimage_0101_best_wk.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0


# echo 'Entrenamientos ... 1111 ... hornet_ + Mopth2 .. 0'

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 1 1 --dump ./runs_models/last_dance/hornet_1111_0/hornet_1111.pth --load_model ./runs_models/last_dance/hornet_1111_0/hornet_1111.pth --version 0 --warm_up 10 --save_per_epoch 30 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_1111_0/hornet_1111_best_acc.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_1111_0/hornet_1111_best_aa.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_1111_0/hornet_1111_best_wk.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0


# echo 'Entrenamientos ... 1001 ... hornet_ + Mopth2 .. 1'

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 0 1 --dump ./runs_models/last_dance/hornet_1001_1/hornet_1001.pth --version 1 --warm_up 10 --save_per_epoch 30 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_1001_1/hornet_0011_best_acc.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_1001_1/hornet_0011_best_aa.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_1001_1/hornet_0011_best_wk.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0


# echo 'Entrenamientos ... 0011 ... hornet_ + Mopth2 .. 2'

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/last_dance/hornet_0011_2/hornet_0011.pth --version 2 --warm_up 10 --save_per_epoch 30 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_0011_2/hornet_0011_best_acc.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_0011_2/hornet_0011_best_aa.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_0011_2/hornet_0011_best_wk.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0



# echo 'Entrenamientos ... 0000 ... hornet_ + Mopth2 .. 3'

# python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 0  --model hornet_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/last_dance/hornet_0000_3/hornet_0000.pth --version 3 --warm_up 10 --save_per_epoch 30 --loss_sensitive --loss_mode 3
# # Acc
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_0000_3/hornet_0000_best_acc.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # AA
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_0000_3/hornet_0000_best_aa.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0
# # WK
# python3.8 main.py --eval --load_model ./runs_models/last_dance/hornet_0000_3/hornet_0000_best_wk.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 0




# Hornet

# Acc
contador=0
for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/hornet_"$i"_"$contador"/hornet_"$i"_best_acc.pth --msg JSONFiles/finales/ddr_hornet_acc/DDR_train.json --device 0 --tipo_head att_h
     let contador++
done;

contador1=0
for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/hornet_"$i"_mopth2_"$contador1"/hornet_"$i"_best_acc.pth --msg JSONFiles/finales/ddr_hornet_acc/DDR_valid.json --device 0 --tipo_head att_h
     let contador1++
done;

contador2=0
for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/hornet_"$i"_mopth2_"$contador2"/hornet_"$i"_best_acc.pth --msg JSONFiles/finales/ddr_hornet_acc/DDR_test.json --device 0 --tipo_head att_h
     let contador2++
done;

# AA

contador3=0
for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/hornet_"$i"_mopth2_"$contador3"/hornet_"$i"_best_aa.pth --msg JSONFiles/finales/ddr_hornet_aa/DDR_train.json --device 0 --tipo_head att_h
     let contador3++
done;

contador4=0
for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/hornet_"$i"_mopth2_"$contador4"/hornet_"$i"_best_aa.pth --msg JSONFiles/finales/ddr_hornet_aa/DDR_valid.json --device 0  --tipo_head att_h
     let contador4++
done;

contador5=0
for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/hornet_"$i"_mopth2_"$contador5"/hornet_"$i"_best_aa.pth --msg JSONFiles/finales/ddr_hornet_aa/DDR_test.json --device 0 --tipo_head att_h
     let contador5++
done;

# WK

contador6=0
for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/hornet_"$i"_mopth2_"$contador6"/hornet_"$i"_best_wk.pth --msg JSONFiles/finales/ddr_hornet_wk/DDR_train.json --device 0 --tipo_head att_h
     let contador6++
done;

contador7=0
for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/hornet_"$i"_mopth2_"$contador7"/hornet_"$i"_best_wk.pth --msg JSONFiles/finales/ddr_hornet_wk/DDR_valid.json --device 0 --tipo_head att_h
     let contador7++
done;

contador8=0
for i in 0000 0001 0011 0101 0111 1001 1011 1101 1111; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/hornet_"$i"_mopth2_"$contador8"/hornet_"$i"_best_wk.pth --msg JSONFiles/finales/ddr_hornet_wk/DDR_test.json --device 0 --tipo_head att_h
     let contador8++
done;