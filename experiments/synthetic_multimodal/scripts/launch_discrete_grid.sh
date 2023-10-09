model_name=(gnf_base)
nf_steps=(1 5 10)
hidden_width=(2000)
hidden_depth=(3 4)
lrs=(1e-3 1e-4)
n_width=(500 250)
n_depth=(4 6)
n_size=(25 50)
sched=(plateau)

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
                    for nw in ${n_width[@]}
                    do
                        for nd in ${n_depth[@]}
                        do
                            for ns in ${n_size[@]}
                            do
                                for s in ${sched[@]}
                                do
                                    sbatch slurm_run_disc_exp.sh $mn $nfs $w $d $lr $nw $nd $ns $s
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
