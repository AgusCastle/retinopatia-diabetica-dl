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
for i in 0111 1001 1011 1101 1111; do
    python3.8 main.py --eval --load_model /home/bringascastle/Documentos/repos/dr-demo/checkpoints/convnext_"$i".pth --dataloader_json JSONFiles/DDR/DDR_  --set valid --att --device 1
    python3.8 main.py --eval --load_model /home/bringascastle/Documentos/repos/dr-demo/checkpoints/convnext_"$i".pth --dataloader_json JSONFiles/DDR/DDR_  --set test --att --device 1
done;