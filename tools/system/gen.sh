#!/bin/sh

for v in $(seq 1 100); do
    invoke check --dataset=digits --what=genetic --filename=model_E.pkl --out="resgen/$v.png" --layer-name=conv3 --nb-iter=100
done
