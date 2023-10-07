model_name=(ffjord_weilbach ffjord_baseline ffjord_strode)
nf_steps=(1 5 10)
hidden_width=(200 500 1000)
hidden_depth=(2 3 8)
lrs=(5e-3 1e-3 1e-4)

for mn in ${model_name[@]}
do
    for nfs in ${nf_steps[@]}
    do
        for w in ${hidden_width[@]}
        do
            for d in ${hidden_depth[@]}
            do
                for lr in ${lrs[@]}
                do
                    sbatch slurm_run_exp.sh $mn $nfs $w $d $lr
                done
            done
        done
    done
done
