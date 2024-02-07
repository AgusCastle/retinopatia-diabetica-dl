# InterImage

# Acc
contador=0
for i in 1111 1001 0011 0000 0001; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/hornet_"$i"_"$contador"/hornet_"$i"_best_acc.pth --msg JSONFiles/last_dance/eyepacs_hornet_acc/eyepacs_train.json --device 1 --tipo_head att_h
     let contador++
done;

contador1=0
for i in 1111 1001 0011 0000 0001; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/hornet_"$i"_"$contador1"/hornet_"$i"_best_acc.pth --msg JSONFiles/last_dance/eyepacs_hornet_acc/eyepacs_valid.json --device 1 --tipo_head att_h
     let contador1++
done;

contador2=0
for i in 1111 1001 0011 0000 0001; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/hornet_"$i"_"$contador2"/hornet_"$i"_best_acc.pth --msg JSONFiles/last_dance/eyepacs_hornet_acc/eyepacs_test.json --device 1 --tipo_head att_h
     let contador2++
done;

# AA

contador3=0
for i in 1111 1001 0011 0000 0001; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/hornet_"$i"_"$contador3"/hornet_"$i"_best_aa.pth --msg JSONFiles/last_dance/eyepacs_hornet_aa/eyepacs_train.json --device 1 --tipo_head att_h
     let contador3++
done;

contador4=0
for i in 1111 1001 0011 0000 0001; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/hornet_"$i"_"$contador4"/hornet_"$i"_best_aa.pth --msg JSONFiles/last_dance/eyepacs_hornet_aa/eyepacs_valid.json --device 1  --tipo_head att_h
     let contador4++
done;

contador5=0
for i in 1111 1001 0011 0000 0001; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/hornet_"$i"_"$contador5"/hornet_"$i"_best_aa.pth --msg JSONFiles/last_dance/eyepacs_hornet_aa/eyepacs_test.json --device 1 --tipo_head att_h
     let contador5++
done;

# WK

contador6=0
for i in 1111 1001 0011 0000 0001; do
     python3.8 main.py --matrix --set train --load_model runs_models/last_dance/hornet_"$i"_"$contador6"/hornet_"$i"_best_wk.pth --msg JSONFiles/last_dance/eyepacs_hornet_wk/eyepacs_train.json --device 1 --tipo_head att_h
     let contador6++
done;

contador7=0
for i in 1111 1001 0011 0000 0001; do
     python3.8 main.py --matrix --set valid --load_model runs_models/last_dance/hornet_"$i"_"$contador7"/hornet_"$i"_best_wk.pth --msg JSONFiles/last_dance/eyepacs_hornet_wk/eyepacs_valid.json --device 1 --tipo_head att_h
     let contador7++
done;

contador8=0
for i in 1111 1001 0011 0000 0001; do
     python3.8 main.py --matrix --set test --load_model runs_models/last_dance/hornet_"$i"_"$contador8"/hornet_"$i"_best_wk.pth --msg JSONFiles/last_dance/eyepacs_hornet_acc/eyepacs_test.json --device 1 --tipo_head att_h
     let contador8++
done;