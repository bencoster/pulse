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

# run
echo "===> run"

for param0 in ${param0Array[*]}
do
    for param1 in ${param1Array[*]}
    do
        for param2 in ${param2Array[*]}
        do
            echo running ... $param0 $param1 $param2 $param3
            python run.py -loss_str "$param0*L2+0$param1*GEOCROSS+0$param2*PERCEPTUAL" \
                        -output_dir ./runs_synthesis \
                        -steps 200

            echo running ... $param0 $param1 $param2 $param3
            python run.py -loss_str "$param0*L2+0$param1*GEOCROSS+0$param2*PERCEPTUAL" \
                        -tile_latent \
                        -output_dir ./runs_synthesis \
                        -steps 200
        done
    done
done

# python run.py -loss_str "100*L2+0.05*GEOCROSS+0.2*PERCEPTUAL"
