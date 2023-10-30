models=(ffjord_baseline_best ffjord_strode_best ffjord_weilbach_best)
for m in ${models[@]}
do
    sbatch evaluate.sh $m
done
