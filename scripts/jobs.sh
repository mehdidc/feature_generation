#!/bin/sh
v=multiplerunsfittest_general.py
m=sparseautoencoder
#sbatch launch python $v --sel=1 --crossover=1 --mutation=0 --layer_name=input --perc=0.1 --initial_source=random --model=$m
#sbatch launch python $v --sel=1 --crossover=1 --mutation=1 --layer_name=input --perc=0.1 --initial_source=random --model=$m

set -x

#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=conv3 --perc=0.1 --initial_source=random --model=$m
#        done
#    done
#done

#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=conv2 --perc=0.1 --initial_source=random --model=$m
#        done
#    done
#done

m=denoisingautoencoder

#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random
#        done
#    done
#done

#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random --model=$m
#        done
#    done
#done

m=sparsedenoisingautoencoder
#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random --model=$m
#        done
#    done
#done


m=sparseautoencoderfonts
#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random --model=$m
#        done
#    done
#done

m=sparseautoencoderflaticons
#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random --model=$m
#        done
#    done
#done

m=nonsparseautoencoder

#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random --model=$m
#        done
#    done
#done


m=sparseautoencoder

#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.4 --initial_source=random --model=$m
#        done
#    done
#done

m=denoisingautoencoder

#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.4 --initial_source=random --model=$m
#        done
#    done
#done


m=denoisingautoencoder

#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random --model=$m
#        done
#    done
#done

m=sparsedenoisingautoencoder
#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random --model=$m
#        done
#    done
#done


m=sparseautoencoder

#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random --model=$m--sort=0
#        done
#    done
#done

#post-iccc

#sbatch launch python $v --sel=0 --crossover=0 --mutation=0 --layer_name=input --perc=0.1 --initial_source=random --model=sparseautoencoder --nb_runs=1000
#sbatch launch python $v --sel=0 --crossover=0 --mutation=0 --layer_name=input --perc=0.1 --initial_source=random --model=sparsedenoisingautoencoder --nb_runs=1000
#sbatch launch python $v --sel=0 --crossover=0 --mutation=0 --layer_name=input --perc=0.1 --initial_source=random --model=denoisingautoencoder --nb_runs=1000

#m=walkbackdenoisingautoencoder
#for cv in 0 1; do
#    for mut in 0 1; do
#        for sel in 0 1; do
#            sbatch launch python $v --sel="$sel" --crossover="$cv" --mutation="$mut" --layer_name=input --perc=0.1 --initial_source=random --model=$m --sort=1
#        done
#    done
#done
