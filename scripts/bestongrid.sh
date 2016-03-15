#!/bin/sh
shuf() { perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@"; }
i=0
rm *.png
for v  in *; do i=$((i+1));cp "$v""/0.png" "$i"".png"; done
#montage -geometry 1:1 $(ls *.png|shuf|head -n 36) out.png
montage -geometry 1:1 *.png out.png
