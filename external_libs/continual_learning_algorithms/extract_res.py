from six.moves import cPickle as pickle

with open('./ring.pickle', 'rb') as f:
	data = pickle.load(f)['mean']

print(data.shape)
print(data[0][-1][-1])
