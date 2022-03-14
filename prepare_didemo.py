import json
import io
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

word2vec = KeyedVectors.load_word2vec_format('./glove_model.bin', binary=True)

j = json.load(open('./data/didemo/val_data.json'))

res = []
for i in j:
	sent = i['description']
	sent = sent.replace(',', '')
	sent = sent.replace('.', '')
	sent = sent.replace('?', '')
	sent = sent.replace('!', '')
	flag = False
	for aa in sent.split():
		try:
			b = word2vec[aa]
		except:
			flag = True
			break
	if flag:
		continue

	r = []
	r.append(i['video'])
	r.append(i['num_segments']*5.)
	time = max(i['times'], key=i['times'].count)
	r.append([time[0]*5., (time[1]+1)*5.])
	sent = i['description']
	r.append(sent)
	sent = sent.replace(',', '')
	sent = sent.replace('.', '')
	sent = sent.replace('?', '')
	sent = sent.replace('!', '')
	r.append(sent.split())
	r.append([k for k in range(len(sent.split()))])
	res.append(r)

with open('./data/didemo/val_new.json','w') as f_obj:
	json.dump(res,f_obj)