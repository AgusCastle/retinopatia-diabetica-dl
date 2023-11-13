# python3.8 main.py --train --epochs 100 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000008 --decay_lr 0.8 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs7/hornet_0001_2/hornet_0001.pth --version 9
# python3.8 main.py --train --epochs 15 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000004 --decay_lr 0.2 --patience 3 --att --attn_block 0 0 1 1 --dump ./runs_models/runs7/hornet_0011_2/hornet_0011.pth --version 10
# python3.8 main.py --train --epochs 15 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000004 --decay_lr 0.2 --patience 3 --att --attn_block 0 1 0 1 --dump ./runs_models/runs7/hornet_0101_2/hornet_0101.pth --version 11
# python3.8 main.py --train --epochs 15 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000004 --decay_lr 0.2 --patience 3 --att --attn_block 0 1 1 1 --dump ./runs_models/runs7/hornet_0111_2/hornet_0111.pth --version 12
# python3.8 main.py --train --epochs 15 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000004 --decay_lr 0.2 --patience 3 --att --attn_block 1 0 0 1 --dump ./runs_models/runs7/hornet_1001_2/hornet_1001.pth --version 13
# python3.8 main.py --train --epochs 15 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000004 --decay_lr 0.2 --patience 3 --att --attn_block 1 0 1 1 --dump ./runs_models/runs7/hornet_1011_2/hornet_1011.pth --version 14
# python3.8 main.py --train --epochs 15 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000004 --decay_lr 0.2 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs7/hornet_1101_2/hornet_1101.pth --version 15
# python3.8 main.py --train --epochs 15 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000004 --decay_lr 0.2 --patience 3 --att --attn_block 1 1 1 1 --dump ./runs_models/runs7/hornet_1111_2/hornet_1111.pth --version 16

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000003 --decay_lr 0.9 --patience 3 --att --attn_block 1 1 0 1 --dump ./runs_models/runs8/hornet_1101_1/hornet_1101.pth --version 10
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.9 --patience 5 --att --attn_block 1 1 0 1 --dump ./runs_models/runs8/hornet_1101_2/hornet_1101.pth --version 20
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000005 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs8/hornet_1101_3/hornet_1101.pth --version 30
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000001 --decay_lr 0.8 --patience 5 --att --attn_block 1 1 0 1 --dump ./runs_models/runs8/hornet_1101_4/hornet_1101.pth --version 40

# python3.8 main.py --eval --load_model ./runs_models/runs7/hornet_1101/hornet_1101.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# python3.8 main.py --eval --load_model ./runs_models/runs7/hornet_1101/hornet_1101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1

# python3.8 main.py --eval --load_model runs_models/runs8/hornet_1101_2/hornet_1101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# python3.8 main.py --eval --load_model runs_models/runs8/hornet_1101_2/hornet_1101.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1

# python3.8 main.py --eval --load_model runs_models/runs8/hornet_1101_3/hornet_1101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# python3.8 main.py --eval --load_model runs_models/runs8/hornet_1101_3/hornet_1101.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1

# python3.8 main.py --eval --load_model runs_models/runs8/hornet_1101_4/hornet_1101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# python3.8 main.py --eval --load_model runs_models/runs8/hornet_1101_4/hornet_1101.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# for i in 0111 1001 1011 1101 1111; do
#     python3.8 main.py --eval --load_model /home/bringascastle/Documentos/repos/dr-demo/checkpoints/convnext_"$i".pth --dataloader_json JSONFiles/DDR/DDR_  --set valid --att --device 1
#     python3.8 main.py --eval --load_model /home/bringascastle/Documentos/repos/dr-demo/checkpoints/convnext_"$i".pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# done;

# Entrenamientos de los modelos
# echo 'Entrenamientos ... 0000'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/runs14/convnext_0000_m_opth_2_60_ce_agabor_0/convnext_0000.pth --version 0 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0000_m_opth_2_60_ce_agabor_0/convnext_0000_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# echo 'Entrenamientos ... 0001'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1 --load_model ./runs_models/runs14/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_0001.pth --dump ./runs_models/runs14/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_0001.pth --version 1 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_small_0001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# echo 'Entrenamientos ... 0011'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --load_model ./runs_models/runs14/convnext_0011_m_opth_2_60_ce_agabor_2/convnext_0011.pth --dump ./runs_models/runs14/convnext_0011_m_opth_2_60_ce_agabor_2/convnext_0011.pth --version 2 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0011_m_opth_2_60_ce_agabor_2/convnext_0011_small__best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# echo 'Entrenamientos ... 0101'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 0 1 --load_model ./runs_models/runs14/convnext_0101_m_opth_2_60_ce_agabor_3/convnext_0101.pth --dump ./runs_models/runs14/convnext_0101_m_opth_2_60_ce_agabor_3/convnext_0101.pth --version 3 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0101_m_opth_2_60_ce_agabor_3/convnext_small_0101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# echo 'Entrenamientos ... 0111'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 1 1 1 --load_model ./runs_models/runs14/convnext_0111_m_opth_2_60_ce_agabor_4/convnext_0111.pth --dump ./runs_models/runs14/convnext_0111_m_opth_2_60_ce_agabor_4/convnext_0111.pth --version 4 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0111_m_opth_2_60_ce_agabor_4/convnext_small_0111_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# echo 'Entrenamientos ... 1001'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 0 1 --load_model ./runs_models/runs14/convnext_1001_m_opth_2_60_ce_agabor_5/convnext_1001.pth --dump ./runs_models/runs14/convnext_1001_m_opth_2_60_ce_agabor_5/convnext_1001.pth --version 5 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_1001_m_opth_2_60_ce_agabor_5/convnext_small_1001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# echo 'Entrenamientos ... 1011'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 0 1 1 --dump ./runs_models/runs14/convnext_1011_m_opth_2_60_ce_agabor_6/convnext_1011.pth --version 6 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_1011_m_opth_2_60_ce_agabor_6/convnext_small_1011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# echo 'Entrenamientos ... 1101'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 0 1 --dump ./runs_models/runs14/convnext_1101_m_opth_2_60_ce_agabor_7/convnext_1101.pth --version 7 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_1101_m_opth_2_60_ce_agabor_7/convnext_small_1101_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# echo 'Entrenamientos ... 1111'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 1 1 --dump ./runs_models/runs14/convnext_1111_m_opth_2_60_ce_agabor_8/convnext_1111.pth --version 8 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_1111_m_opth_2_60_ce_agabor_8/convnext_small_1111_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1


# La SNF va a ssalir mal pero yo sigo al pie de la letra 

# InternImage

# python3.8 main.py --snf_train --lr 1.2 --decay_lr 0.6 --patience 100 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_12_2.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_19_2.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --snf_train --lr 0.3 --decay_lr 0.9 --patience 300 --epochs 800 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_5.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 1.3 --decay_lr 0.9 --patience 200 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_7.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 2.2 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF3/SNF_8.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 2.1 --decay_lr 0.8 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_9.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 0.9 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF3/SNF_10.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 0.9 --decay_lr 0.7 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_11.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 1.5 --decay_lr 0.6 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF3/SNF_12.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --snf_train --lr 1.5 --decay_lr 0.6 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_13.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 1.2 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF3/SNF_14.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 1.2 --decay_lr 0.7 --patience 90 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_15.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF3/SNF_16.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_17.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.9 --patience 80 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF3/SNF_18.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF3/SNF_19.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 1 1 1 1 --dump ./runs_models/runs13/hornet_1111_m_opth_2_60_ce_agabor_8/hornet_1111.pth --version 8 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs13/hornet_1111_m_opth_2_60_ce_agabor_8/hornet_1111_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1

# Hornet

# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H --snf_train --lr 1.2 --decay_lr 0.6 --patience 100 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_12_2_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_19_2_H.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H --snf_train --lr 0.3 --decay_lr 0.9 --patience 300 --epochs 800 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_5_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H --snf_train --lr 1.3 --decay_lr 0.9 --patience 200 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_7_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 2.2 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF4/SNF_8_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 2.1 --decay_lr 0.8 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_9_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 0.9 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF4/SNF_10_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 0.9 --decay_lr 0.7 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_11_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 1.5 --decay_lr 0.6 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF4/SNF_12_H.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 1.5 --decay_lr 0.6 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_13_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 1.2 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF4/SNF_14_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 1.2 --decay_lr 0.7 --patience 90 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_15_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF4/SNF_16_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_17_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 1.1 --decay_lr 0.9 --patience 80 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF4/SNF_18_H.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_H  --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF4/SNF_19_H.pth --loss_sensitive --loss_mode 3

# # ConvNext

# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.2 --decay_lr 0.6 --patience 100 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_12_2_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_19_2_C.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 0.3 --decay_lr 0.9 --patience 300 --epochs 800 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_5_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.3 --decay_lr 0.9 --patience 200 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_7_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 2.2 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF5/SNF_8_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C --snf_train --lr 2.1 --decay_lr 0.8 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_9_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 0.9 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF5/SNF_10_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 0.9 --decay_lr 0.7 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_11_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.5 --decay_lr 0.6 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF5/SNF_12_C.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.5 --decay_lr 0.6 --patience 80 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_13_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.2 --decay_lr 0.7 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF5/SNF_14_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.2 --decay_lr 0.7 --patience 90 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_15_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF5/SNF_16_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.1 --decay_lr 0.8 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_17_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.1 --decay_lr 0.9 --patience 80 --epochs 400 --batch 128 --device 1 --dump ./runs_models/runSNF5/SNF_18_C.pth --loss_sensitive --loss_mode 3
# python3.8 main.py --dataloader_json JSONFiles/DDR_M_C  --snf_train --lr 1.1 --decay_lr 0.9 --patience 40 --epochs 400 --batch 256 --device 1 --dump ./runs_models/runSNF5/SNF_19_C.pth --loss_sensitive --loss_mode 3

# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0000_m_opth_2_60_ce_agabor_0/convnext_small_0000_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_small_0001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0011_m_opth_2_60_ce_agabor_2/convnext_small_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1

# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0000_m_opth_2_60_ce_agabor_0/convnext_small_0000_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set valid --att --device 1
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_small_0001_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set valid --att --device 1
# python3.8 main.py --eval --load_model ./runs_models/runs14/convnext_0011_m_opth_2_60_ce_agabor_2/convnext_small_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set valid --att --device 1


# echo 'Entrenamientos convnext ... 0001 con eyepacs'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.00002 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 1  --dump ./runs_models/runs15/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_0001.pth --version 1 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs15/convnext_0001_m_opth_2_60_ce_agabor_1/convnext_small_0001_best.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test --att --device 1

# echo 'Entrenamientos convnext ... 0000'
# python3.8 main.py --train --epochs 30 --batch 8 --workers 8 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --lr 0.00002 --decay_lr 0.3 --patience 3 --att --attn_block 0 0 0 0 --dump ./runs_models/runs15/convnext_0000_m_opth_2_60_ce_agabor_0/convnext_0000.pth --version 0 --loss_sensitive --loss_mode 3 --base_loss ce --gabor 2 --no_g 1 
# python3.8 main.py --eval --load_model ./runs_models/runs15/convnext_0000_m_opth_2_60_ce_agabor_0/convnext_small_0000_best.pth --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_  --set test --att --device 1

# echo 'Entrenamientos ... 0011'
# python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 0  --model internimage_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 1 1 --dump ./runs_models/runs15/internimage_0011_m_opth_2_60_ce_agabor_ultimo_2/internimage_0011.pth --version 2 --loss_sensitive --loss_mode 3 --base_loss ce
# python3.8 main.py --eval --load_model ./runs_models/runs15/internimage_0011_m_opth_2_60_ce_agabor_ultimo_2/internimage_0011_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 0

#echo 'Entrenamientos ... 0000 ... HorNet'
#python3.8 main.py --train --epochs 60 --batch 4 --workers 4 --device 1  --model hornet_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/runs16/hornet_0000_ce_1/hornet_0000.pth --version 1
python3.8 main.py --eval --load_model ./runs_models/runs16/hornet_0000_ce_1/hornet_0000_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
 
#echo 'Entrenamientos ... 0000 ... ConvNeXt'
python3.8 main.py --train --epochs 30 --batch 4 --workers 4 --device 1  --model convnext_small_ --dataloader_json ./JSONFiles/DDR/DDR_ --lr 0.000007 --decay_lr 0.8 --patience 8 --att --attn_block 0 0 0 0 --dump ./runs_models/runs16/convnext_0000_ce_2/convnext_0000.pth --version 2  
python3.8 main.py --eval --load_model ./runs_models/runs16/convnext_0000_ce_2/convnext_small_0000_best.pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
 