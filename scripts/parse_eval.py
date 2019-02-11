import re, sys, os
import matplotlib.pyplot as plt
from matplotlib._color_data import XKCD_COLORS

with open(sys.argv[1], 'r') as f:
	txt = f.read()

txt, overall = txt.split('overall performance')

class_names = []
mAP_overall = []
mAP_small   = []
mAP_medium  = []
mAP_large   = []

for class_result in txt.split('evaluate category: ')[1:]:
	lines = class_result.split('\n')
	class_names.append(lines[0])

	def grabMAP(string):
		return float(string.split('] = ')[1]) * 100
	
	mAP_overall.append(grabMAP(lines[ 7]))
	mAP_small  .append(grabMAP(lines[10]))
	mAP_medium .append(grabMAP(lines[11]))
	mAP_large  .append(grabMAP(lines[12]))

mAP_map = {
	'small': mAP_small,
	'medium': mAP_medium,
	'large': mAP_large,
}

if len(sys.argv) > 2:
	bars = plt.bar(class_names, mAP_map[sys.argv[2]])
	plt.title(sys.argv[2] + ' mAP per class')
else:
	bars = plt.bar(class_names, mAP_overall)
	plt.title('overall mAP per class')

colors = list(XKCD_COLORS.values())

for idx, bar in enumerate(bars):
	# Mmm pseudorandom colors
	char_sum = sum([ord(char) for char in class_names[idx]])
	bar.set_color(colors[char_sum % len(colors)])

plt.xticks(rotation='vertical')
plt.show()
