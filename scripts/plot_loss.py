import re, sys, os
import matplotlib.pyplot as plt

from utils.functions import MovingAverage

with open(sys.argv[1], 'r') as f:
	inp = f.read()

pattern = re.compile(r'\[\s*(?P<epoch>\d+)\]\s*(?P<iteration>\d+) \|\| B: (?P<b>\S+) \| C: (?P<c>\S+) \| M: (?P<m>\S+) \| T: (?P<t>\S+)')
data = []

for line in inp.split('\n'):
	f = pattern.search(line)
	
	if f is not None:
		datum = f.groupdict()
		for k, v in datum.items():
			datum[k] = float(v)
		
		if not 'Validation' in line:
			data.append(datum)


def smoother(y, interval=100):
	avg = MovingAverage(interval)

	for i in range(len(y)):
		avg.append(y[i])
		y[i] = avg.get_avg()
	
	return y

plt.title(os.path.basename(sys.argv[1]))
plt.xlabel('Iteration')
plt.ylabel('Loss')

x = [x['iteration'] for x in data]
plt.plot(x, smoother([y['b'] for y in data]))
plt.plot(x, smoother([y['c'] for y in data]))
plt.plot(x, smoother([y['m'] for y in data]))

plt.legend(['BBox Loss', 'Conf Loss', 'Mask Loss'])
plt.show()
