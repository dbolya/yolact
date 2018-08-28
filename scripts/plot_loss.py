import re, sys, os
import matplotlib.pyplot as plt

from utils.functions import MovingAverage

with open(sys.argv[1], 'r') as f:
	inp = f.read()

patterns = {
	'train': re.compile(r'\[\s*(?P<epoch>\d+)\]\s*(?P<iteration>\d+) \|\| B: (?P<b>\S+) \| C: (?P<c>\S+) \| M: (?P<m>\S+) \| T: (?P<t>\S+)'),
	'val': re.compile(r'\s*(?P<type>[a-z]+) \|\s*(?P<all>\S+)')
}
data = {key: [] for key in patterns}

for line in inp.split('\n'):
	for key, pattern in patterns.items():
		f = pattern.search(line)
		
		if f is not None:
			datum = f.groupdict()
			for k, v in datum.items():
				try:
					v = float(v)
				except ValueError:
					pass
				datum[k] = v
			
			data[key].append(datum)
			break


def smoother(y, interval=100):
	avg = MovingAverage(interval)

	for i in range(len(y)):
		avg.append(y[i])
		y[i] = avg.get_avg()
	
	return y

def plot_train(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Training Loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	x = [x['iteration'] for x in data]
	plt.plot(x, smoother([y['b'] for y in data]))
	plt.plot(x, smoother([y['c'] for y in data]))
	plt.plot(x, smoother([y['m'] for y in data]))

	plt.legend(['BBox Loss', 'Conf Loss', 'Mask Loss'])
	plt.show()

def plot_val(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Validation mAP')
	plt.xlabel('Idx')
	plt.ylabel('mAP')

	x = range(len(data) // 2)
	plt.plot(x, [x['all'] for x in data if x['type'] == 'box'])
	plt.plot(x, [x['all'] for x in data if x['type'] == 'mask'])

	plt.legend(['BBox mAP', 'Mask mAP'])
	plt.show()

if len(sys.argv) > 2 and sys.argv[2] == 'val':
	plot_val(data['val'])
else:
	plot_train(data['train'])
