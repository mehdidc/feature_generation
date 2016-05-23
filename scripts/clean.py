
if __name__ == '__main__':
	from lightjob.cli import load_db
	from lightjob.db import SUCCESS
	import shutil
	import os
	db = load_db()
	def rm(path):
		if not os.path.exists(path):
			return
		print('Removing {}'.format(path))
		if os.path.isdir(path):
			shutil.rmtree(path)
		else:
			os.remove(path)

	for j in db.jobs_with(state=SUCCESS, type="generation"):
		j = dict(j)
		s = j['content']['model_summary']
 		ref_job = db.get_job_by_summary(s)
		model_details = ref_job['content']
 		# training job
		folder = 'jobs/results/{}'.format(ref_job['summary'])
		#rm(os.path.join(folder, 'model.pkl'))
		rm(os.path.join(folder, 'features'))
		rm(os.path.join(folder, 'recons'))
		rm(os.path.join(folder, 'out'))
		# generation job of the training job
		folder = 'jobs/results/{}'.format(j['summary'])
		rm(os.path.join(folder, 'iterations'))