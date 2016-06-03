from lightjob.cli import load_db
import click
import json
import pandas as pd

@click.group()
def main():
    pass

@click.command()
@click.option('--action', help='action', required=True)
@click.option('--where', default=None, help='where', required=False)
@click.option('--type', default=None, help='where', required=False)
@click.option('--ref_where', default=None, help='ref_where', required=False)
@click.option('--state', default=None, help='state', required=False)
@click.option('--dontcare', default=False, help='dont care if delete', required=False)
@click.option('--details', default=False, help='show details', required=False)
def do(action, where, type, ref_where, state, dontcare, details):
	db = load_db()
	kw = {}
	if where:
		kw['where'] = where
	if type:
		kw['type'] = type
	if state:
		kw['state'] = state
	def accepted(job):
		ok = True
		if ref_where:
			jref = db.get_job_by_summary(job['content']['model_summary'])
			ok = ok and jref['where'] == ref_where
		return ok
	jobs = db.jobs_with(**kw)
	jobs = filter(accepted, jobs)

	def get_key(j):

		if 'life' in j and j['life'] is not None:
			l = j['life'][-1]['dt']
			l = pd.to_datetime(l, infer_datetime_format=True)
			return l
		else:
			return pd.to_datetime('Fri Jan 01 00:00:40 1970')


	jobs = sorted(jobs, key=get_key)
	if action == 'show':
		for j in jobs:
			j = dict(j)
			if details:
				print(json.dumps(j, indent=4))
			else:
				print(j['summary'])
	if action == 'delete':
		if dontcare is None:
			print('Please specify that you dont care if you want to delete...')
			return
		if dontcare == False:
			print('ok you care, so I will not delete anything...')
			return
		for j in jobs:
			print('Deleting {}'.format(j['summary']))
			db.delete(dict(summary=j['summary']))
	print('Total nb of jobs processed : {}'.format(len(jobs)))

if __name__ == '__main__':
	main.add_command(do)
	main()
    