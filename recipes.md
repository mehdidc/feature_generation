1) invoke check --what=genetic --dataset=digits --params=json/genetic.json --filename=models/model_E.pkl
2) invoke train --dataset=digits --model-name=model46 --prefix=training/46digits
3) invoke check --what=simple_genetic --dataset=digits --params=json/simple_genetic.json --filename=models/model_E.pkl
4) python run_jobs.py --force=1 -l $(grep Error jobs/outputs/*|awk 'BEGIN{FS=":"}{print $1}'| sed -e "s/outputs/running/g")
