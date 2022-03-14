import json
import io
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

word2vec = KeyedVectors.load_word2vec_format('./glove_model.bin', binary=True)

j = json.load(open('./data/charades/train.json'))

res = []
count = 0
for i in j.keys():
	r = []
	r.append(i)
	r.append(j[i]['video_duration'])
	for m in range(len(j[i]['timestamps'])):
		a = j[i]['sentences'][m].split()
		flag = False
		for aa in a:
			try:
				b = word2vec[aa]
			except:
				flag = True
				break
		if flag:
			continue

		r.append(j[i]['timestamps'][m])
		r.append(j[i]['sentences'][m])
		r.append(j[i]['sentences'][m].split())
		r.append([k for k in range(len(j[i]['sentences'][m].split()))])
		# if i == 'HL9YR':
		# print(r)
		# 	print(len(r))
		res.append(r)
		count += 1
		# print(r)
		r = []
		r.append(i)
		r.append(j[i]['video_duration'])
print(count)
with open('./data/charades/train_new.json','w') as f_obj:
	json.dump(res,f_obj)