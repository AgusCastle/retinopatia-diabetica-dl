# Convnext

# Acc
contador=0
for i in 0000 1101 1001 0001 1111; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/convnext_"$i"_"$contador"/convnext_small_"$i"_best_acc.pth --msg JSONFiles/last_dance/eyepacs_convnext_acc/eyepacs_train.json --device 0 --tipo_head att_c
     let contador++
done;

contador1=0
for i in 0000 1101 1001 0001 1111; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/convnext_"$i"_"$contador1"/convnext_small_"$i"_best_acc.pth --msg JSONFiles/last_dance/eyepacs_convnext_acc/eyepacs_valid.json --device 0 --tipo_head att_c
     let contador1++
done;

contador2=0
for i in 0000 1101 1001 0001 1111; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/convnext_"$i"_"$contador2"/convnext_small_"$i"_best_acc.pth --msg JSONFiles/last_dance/eyepacs_convnext_acc/eyepacs_test.json --device 0 --tipo_head att_c
     let contador2++
done;

# AA

contador3=0
for i in 0000 1101 1001 0001 1111; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/convnext_"$i"_"$contador3"/convnext_small_"$i"_best_aa.pth --msg JSONFiles/last_dance/eyepacs_convnext_aa/eyepacs_train.json --device 0 --tipo_head att_c
     let contador3++
done;

contador4=0
for i in 0000 1101 1001 0001 1111; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/convnext_"$i"_"$contador4"/convnext_small_"$i"_best_aa.pth --msg JSONFiles/last_dance/eyepacs_convnext_aa/eyepacs_valid.json --device 0  --tipo_head att_c
     let contador4++
done;

contador5=0
for i in 0000 1101 1001 0001 1111; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/convnext_"$i"_"$contador5"/convnext_small_"$i"_best_aa.pth --msg JSONFiles/last_dance/eyepacs_convnext_aa/eyepacs_test.json --device 0 --tipo_head att_c
     let contador5++
done;

# WK

contador6=0
for i in  0000 1101 1001 0001 1111; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/convnext_"$i"_"$contador6"/convnext_small_"$i"_best_wk.pth --msg JSONFiles/last_dance/eyepacs_convnext_wk/eyepacs_train.json --device 0 --tipo_head att_c
     let contador6++
done;

contador7=0
for i in 0000 1101 1001 0001 1111; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/convnext_"$i"_"$contador7"/convnext_small_"$i"_best_wk.pth --msg JSONFiles/last_dance/eyepacs_convnext_wk/eyepacs_valid.json --device 0 --tipo_head att_c
     let contador7++
done;

contador8=0
for i in 0000 1101 1001 0001 1111; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/convnext_"$i"_"$contador8"/convnext_small_"$i"_best_wk.pth --msg JSONFiles/last_dance/eyepacs_convnext_acc/eyepacs_test.json --device 0 --tipo_head att_c
     let contador8++
done;