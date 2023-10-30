seed=(2541 2547 412 411 321 3431 4273 2526)
models=(straf_best gnf_best anf_best)
for m in ${models[@]}
do
    for s in ${seed[@]}
    do
        sbatch run_best_disc.sh $m $s
    done
done


