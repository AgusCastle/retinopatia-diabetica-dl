# InterImage

# Acc
contador=0
for i in 1101 1011 0000 0011 0101; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/internimage_"$i"_"$contador"/internimage_"$i"_best_acc.pth --msg JSONFiles/last_dance/eyepacs_intern_acc/eyepacs_train.json --device 0 --tipo_head att_i
     let contador++
done;

contador1=0
for i in 1101 1011 0000 0011 0101; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/internimage_"$i"_"$contador1"/internimage_"$i"_best_acc.pth --msg JSONFiles/last_dance/eyepacs_intern_acc/eyepacs_valid.json --device 0 --tipo_head att_i
     let contador1++
done;

contador2=0
for i in 1101 1011 0000 0011 0101; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/internimage_"$i"_"$contador2"/internimage_"$i"_best_acc.pth --msg JSONFiles/last_dance/eyepacs_intern_acc/eyepacs_test.json --device 0 --tipo_head att_i
     let contador2++
done;

# AA

contador3=0
for i in 1101 1011 0000 0011 0101; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/internimage_"$i"_"$contador3"/internimage_"$i"_best_aa.pth --msg JSONFiles/last_dance/eyepacs_intern_aa/eyepacs_train.json --device 0 --tipo_head att_i
     let contador3++
done;

contador4=0
for i in 1101 1011 0000 0011 0101; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/internimage_"$i"_"$contador4"/internimage_"$i"_best_aa.pth --msg JSONFiles/last_dance/eyepacs_intern_aa/eyepacs_valid.json --device 0  --tipo_head att_i
     let contador4++
done;

contador5=0
for i in 1101 1011 0000 0011 0101; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/internimage_"$i"_"$contador5"/internimage_"$i"_best_aa.pth --msg JSONFiles/last_dance/eyepacs_intern_aa/eyepacs_test.json --device 0 --tipo_head att_i
     let contador5++
done;

# WK

contador6=0
for i in  1101 1011 0000 0011 0101; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/internimage_"$i"_"$contador6"/internimage_"$i"_best_wk.pth --msg JSONFiles/last_dance/eyepacs_intern_wk/eyepacs_train.json --device 0 --tipo_head att_i
     let contador6++
done;

contador7=0
for i in 1101 1011 0000 0011 0101; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/internimage_"$i"_"$contador7"/internimage_"$i"_best_wk.pth --msg JSONFiles/last_dance/eyepacs_intern_wk/eyepacs_valid.json --device 0 --tipo_head att_i
     let contador7++
done;

contador8=0
for i in 1101 1011 0000 0011 0101; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/internimage_"$i"_"$contador8"/internimage_"$i"_best_wk.pth --msg JSONFiles/last_dance/eyepacs_intern_acc/eyepacs_test.json --device 0 --tipo_head att_i
     let contador8++
done;