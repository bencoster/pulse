# param0 L2
echo "===> Initialize L2"

param0Array=()
start=99
end=100
step=1

tmp=$start
while [ 1 = "$(echo "$tmp < $end" | bc -l)" ]
do
    tmp=$(echo "$tmp + $step" | bc -l)
    param0Array[${#param0Array[*]}]=$tmp
done

# param1 GEOCROSS
echo "===> Initialize GEOCROSS"

param1Array=()
start=0
end=1
step=0.05

tmp=$start
while [ 1 = "$(echo "$tmp < $end" | bc -l)" ]
do
    tmp=$(echo "$tmp + $step" | bc -l)
    param1Array[${#param1Array[*]}]=$tmp
done

# param2 PERCEPTUAL
echo "===> Initialize PERCEPTUAL"

param2Array=()
start=0
end=0.8
step=0.05

tmp=$start
while [ 1 = "$(echo "$tmp < $end" | bc -l)" ]
do
    tmp=$(echo "$tmp + $step" | bc -l)
    param2Array[${#param2Array[*]}]=$tmp
done

# param3 ARC
echo "===> Initialize ARC"

param3Array=()
start=3
end=15
step=1

tmp=$start
while [ 1 = "$(echo "$tmp < $end" | bc -l)" ]
do
    tmp=$(echo "$tmp + $step" | bc -l)
    param3Array[${#param3Array[*]}]=$tmp
done

# param4 num_trainable_noise_layers

param4Array=()
start=5
end=17
step=1

tmp=$start
while [ 1 = "$(echo "$tmp < $end" | bc -l)" ]
do
    tmp=$(echo "$tmp + $step" | bc -l)
    param4Array[${#param4Array[*]}]=$tmp
done

# run
echo "===> run"

for param0 in ${param0Array[*]}
do
    for param1 in ${param1Array[*]}
    do
        for param2 in ${param2Array[*]}
        do
            for param3 in ${param3Array[*]}
            do
                for param4 in ${param4Array[*]}
                do
                    echo "==========================================="
                    echo running ... $param0 $param1 $param2 $param3
                    python run.py -loss_str "$param0*L2+0$param1*GEOCROSS+0$param2*PERCEPTUAL+0$param3*ARC" \
                                -output_dir ./runs_synthesis \
                                -steps 500 \
                                -num_trainable_noise_layers $param4

                    echo "==========================================="
                    echo running ... $param0 $param1 $param2 $param3 -tile_latent
                    python run.py -loss_str "$param0*L2+0$param1*GEOCROSS+0$param2*PERCEPTUAL+0$param3*ARC" \
                                -tile_latent \
                                -output_dir ./runs_synthesis \
                                -steps 500 \
                                -num_trainable_noise_layers $param4
                done
            done
        done
    done
done

# python run.py -loss_str "100*L2+0.05*GEOCROSS+0.2*PERCEPTUAL"
