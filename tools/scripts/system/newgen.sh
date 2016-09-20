#!/bin/sh

#$1 : directory
#$2 : name of layer 
#$3 : start number

for v in $(seq $3 100); do
    invoke check --dataset=digits --what=genetic --filename=model_E.pkl --out="$1/$v.png" --layer-name=$2 --nb-iter=100 2>&1 |tee  "$1/out"
    cat $1/out >> $1/log
done

cd $1
for v in *.png; do 
    convert -trim $v $v
done

montage -geometry +0+0 *.png all.png
