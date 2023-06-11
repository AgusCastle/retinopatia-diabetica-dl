# Bash entrenamientos para Servidor 1 (GPU 0)

python3.8 main.py --train --epochs 30 --lr 0.000007 --decay_lr 0.2 --batch 8 --workers 8 --device 1 --patience 3 --dataloader_json ./JSONFiles/eyepacs_resam/eyepacs_ --dump ./runs4/ConvNeXt_0000_9010_22/ConvNeXt_0000.pth --version 22 --model convnext_custom --mode multi
