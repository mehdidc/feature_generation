#!/bin/sh
#scripts/create 48digits
#sbatch launch invoke train --dataset=digits --model-name=model8 --prefix=training/48digits

#scripts/create 48fonts
#sbatch launch invoke train --dataset=fonts --model-name=model48 --prefix=training/48fonts --force-w=28 --force-h=28

#scripts/create 48digitsTrueOne
#sbatch launch invoke train --dataset=digits --model-name=model48 --prefix=training/48digitsTrueOne

#scripts/create 49fonts28x28
#sbatch launch invoke train --dataset=fonts --model-name=model49 --prefix=training/49fonts28x28 --force-w=28 --force-h=28 --params="json/denoise.json"

#scripts/create 49fonts64x64
#sbatch launch invoke train --dataset=fonts --model-name=model49 --prefix=training/49fonts64x64 --params="json/denoise.json"


#scripts/create 47digits_noise05
#sbatch launch invoke train --dataset=digits --model-name=model47 --prefix=training/47digits_noise05 --params="json/denoise05.json"

#scripts/create 47digits_noise07
#sbatch launch invoke train --dataset=digits --model-name=model47 --prefix=training/47digits_noise07 --params="json/denoise07.json"

#scripts/create 47digits_noise03
#sbatch launch invoke train --dataset=digits --model-name=model47 --prefix=training/47digits_noise03 --params="json/denoise03.json"

#scripts/create 47digits_noise02
#sbatch launch invoke train --dataset=digits --model-name=model47 --prefix=training/47digits_noise02 --params="json/denoise02.json"

#scripts/create 47digits_noise01
#sbatch launch invoke train --dataset=digits --model-name=model47 --prefix=training/47digits_noise01 --params="json/denoise01.json"


#scripts/create 49fonts28x28_walkback
#sbatch launch invoke train --dataset=fonts --model-name=model49 --prefix=training/49fonts28x28_walkback --force-w=28 --force-h=28 --params="json/walkback.json"

#scripts/create 49fonts64x64_walkback
#sbatch launch invoke train --dataset=fonts --model-name=model49 --prefix=training/49fonts64x64_walkback --params="json/walkback.json"

