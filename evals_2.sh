# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set valid --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_valid.json --device 0
# done;

# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set test --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_test.json --device 0
# done;

# for i in convnext_agus convnext_ab_agus convnext_0011 convnext_0101 convnext_0111 convnext_1001 convnext_1011 convnext_1101 convnext_1111; do
#     python3.8 main.py --matrix --set train --load_model runs2/$i/$i.pth --json_result JSONFiles/DDR_M/DDR_train.json --device 0
# done;

python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1001_9010_12/ConvNeXt_1001.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1001_9010_12/ConvNeXt_1001.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1011_9010_16/ConvNeXt_1011.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1011_9010_16/ConvNeXt_1011.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1001_9010_17/ConvNeXt_1001.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1001_9010_17/ConvNeXt_1001.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1001_9010_18/ConvNeXt_1001.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1001_9010_18/ConvNeXt_1001.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test

python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1001_9010_20/ConvNeXt_1001.pth --dataloader_json JSONFiles/eyepacs_resam/eyepacs_ --set test
python3.8 main.py --eval --device 1 --load_model runs4/ConvNeXt_1001_9010_20/ConvNeXt_1001.pth --dataloader_json JSONFiles/eyepacs_resam/messidor2_ --set test
