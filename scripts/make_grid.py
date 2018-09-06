import numpy as np
import math, random

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.24)
im_handle = None

save_path = 'grid.np'

center_x, center_y = (0.5, 0.5)
grid_w, grid_h = (35, 35)
spacing = 0
scale = 4
angle = 0
grid = None

all_grids = []
unique = False

# A hack
disable_render = False

def render():
	if disable_render:
		return

	x = np.tile(np.array(list(range(grid_w)), dtype=np.float).reshape(1, grid_w), [grid_h, 1]) - grid_w * center_x
	y = np.tile(np.array(list(range(grid_h)), dtype=np.float).reshape(grid_h, 1), [1, grid_w]) - grid_h * center_y

	x /= scale
	y /= scale

	a1 =  angle + math.pi / 3
	a2 = -angle + math.pi / 3
	a3 =  angle

	z1 = x * math.sin(a1) + y * math.cos(a1)
	z2 = x * math.sin(a2) - y * math.cos(a2)
	z3 = x * math.sin(a3) + y * math.cos(a3)

	s1 = np.square(np.sin(z1))
	s2 = np.square(np.sin(z2))
	s3 = np.square(np.sin(z3))

	line_1 = np.exp(s1 * spacing) * s1
	line_2 = np.exp(s2 * spacing) * s2
	line_3 = np.exp(s3 * spacing) * s3

	global grid
	grid = np.clip(1 - (line_1 + line_2 + line_3) / 3, 0, 1)

	global im_handle
	if im_handle is None:
		im_handle = plt.imshow(grid)
	else:
		im_handle.set_data(grid)
	fig.canvas.draw_idle()

def update_scale(val):
	global scale
	scale = val

	render()

def update_angle(val):
	global angle
	angle = val

	render()

def update_centerx(val):
	global center_x
	center_x = val

	render()

def update_centery(val):
	global center_y
	center_y = val

	render()

def update_spacing(val):
	global spacing
	spacing = val

	render()

def randomize(val):
	global center_x, center_y, spacing, scale, angle, disable_render

	center_x, center_y = (random.uniform(0, 1), random.uniform(0, 1))
	spacing = random.uniform(-0.2, 2)
	scale = 4 * math.exp(random.uniform(-1, 1))
	angle = random.uniform(-math.pi, math.pi)

	disable_render = True

	scale_slider.set_val(scale)
	angle_slider.set_val(angle)
	centx_slider.set_val(center_x)
	centy_slider.set_val(center_y)
	spaci_slider.set_val(spacing)

	disable_render = False

	render()

def add(val):
	all_grids.append(grid)

	global unique
	if not unique:
		unique = test_uniqueness(np.stack(all_grids))
	
	export_len_text.set_text('Num Grids: ' + str(len(all_grids)))
	fig.canvas.draw_idle()

def add_randomize(val):
	add(val)
	randomize(val)

def export(val):
	np.stack(all_grids).dump(save_path)
	print('Saved %d grids to "%s".' % (len(all_grids), save_path))

	global unique
	unique = False
	all_grids.clear()

	export_len_text.set_text('Num Grids: ' + str(len(all_grids)))
	fig.canvas.draw_idle()

def test_uniqueness(grids):
	# Grids shape [ngrids, h, w]
	grids = grids.reshape((-1, grid_h, grid_w))

	for y in range(grid_h):
		for x in range(grid_h):
			pixel_features = grids[:, y, x]

			# l1 distance for this pixel with every other
			l1_dist = np.sum(np.abs(grids - np.tile(pixel_features, grid_h*grid_w).reshape((-1, grid_h, grid_w))), axis=0)

			# Equal if l1 distance is really small. Note that this will include this pixel
			num_equal = np.sum((l1_dist < 0.0001).astype(np.int32))

			if num_equal > 1:
				print('Pixel at (%d, %d) has %d other pixel%s with the same representation.' % (x, y, num_equal-1, '' if num_equal==2 else 's'))
				return False
	
	print('Each pixel has a distinct representation.')
	return True



render()

axis = plt.axes([0.22, 0.19, 0.59, 0.03], facecolor='lightgoldenrodyellow')
scale_slider = Slider(axis, 'Scale', 0.1, 20, valinit=scale, valstep=0.1)
scale_slider.on_changed(update_scale)

axis = plt.axes([0.22, 0.15, 0.59, 0.03], facecolor='lightgoldenrodyellow')
angle_slider = Slider(axis, 'Angle', -math.pi, math.pi, valinit=angle, valstep=0.1)
angle_slider.on_changed(update_angle)

axis = plt.axes([0.22, 0.11, 0.59, 0.03], facecolor='lightgoldenrodyellow')
centx_slider = Slider(axis, 'Center X', 0, 1, valinit=center_x, valstep=0.05)
centx_slider.on_changed(update_centerx)

axis = plt.axes([0.22, 0.07, 0.59, 0.03], facecolor='lightgoldenrodyellow')
centy_slider = Slider(axis, 'Center Y', 0, 1, valinit=center_y, valstep=0.05)
centy_slider.on_changed(update_centery)

axis = plt.axes([0.22, 0.03, 0.59, 0.03], facecolor='lightgoldenrodyellow')
spaci_slider = Slider(axis, 'Spacing', -1, 2, valinit=spacing, valstep=0.05)
spaci_slider.on_changed(update_spacing)

axis = plt.axes([0.8, 0.54, 0.15, 0.05], facecolor='lightgoldenrodyellow')
rando_button = Button(axis, 'Randomize')
rando_button.on_clicked(randomize)

axis = plt.axes([0.8, 0.48, 0.15, 0.05], facecolor='lightgoldenrodyellow')
addgr_button = Button(axis, 'Add')
addgr_button.on_clicked(add)

# Likely not a good way to do this but whatever
export_len_text = plt.text(0, 3, 'Num Grids: 0')

axis = plt.axes([0.8, 0.42, 0.15, 0.05], facecolor='lightgoldenrodyellow')
addra_button = Button(axis, 'Add / Rand')
addra_button.on_clicked(add_randomize)

axis = plt.axes([0.8, 0.36, 0.15, 0.05], facecolor='lightgoldenrodyellow')
saveg_button = Button(axis, 'Save')
saveg_button.on_clicked(export)



plt.show()
