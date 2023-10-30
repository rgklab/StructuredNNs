seed=(2541 2547 412 411 321 3431 4273 2526)
models=(ffjord_weilbach_best ffjord_baseline_best ffjord_strode_best)
for m in ${models[@]}
do
    for s in ${seed[@]}
    do
        sbatch run_best_cont.sh $m $s
    done
done
