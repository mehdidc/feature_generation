from lightjob.db import TinyDBWrapper, BlitzDBWrapper, migrate
src = TinyDBWrapper()
src.load('.lightjob/db.json')
dst = BlitzDBWrapper()
dst.load('.lightjob/bl')
print(len(src.all_jobs()))
#dst.insert_list(src.all_jobs())
j = dst.all_jobs()
j = list(j)
print(len(j))
src.close()