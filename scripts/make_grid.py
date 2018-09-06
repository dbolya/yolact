import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
im_handle = None

center_x, center_y = (0.5, 0.5)
grid_w, grid_h = (35, 35)
scale = 4
angle = 0

def render():
	x = np.tile(np.array(list(range(grid_w)), dtype=np.float).reshape(1, grid_w), [grid_h, 1]) - grid_w * center_x
	y = np.tile(np.array(list(range(grid_h)), dtype=np.float).reshape(grid_h, 1), [1, grid_w]) - grid_h * center_y

	x /= scale
	y /= scale

	a1 =  angle + math.pi / 3
	a2 = -angle + math.pi / 3
	a3 =  angle

	line_1 = np.square(np.sin(x * math.sin(a1) + y * math.cos(a1)))
	line_2 = np.square(np.sin(x * math.sin(a2) - y * math.cos(a2)))
	line_3 = np.square(np.sin(x * math.sin(a3) + y * math.cos(a3)))

	grid = 1 - (line_1 + line_2 + line_3) / 3

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

render()

axfreq = plt.axes([0.22, 0.15, 0.59, 0.03], facecolor='lightgoldenrodyellow')
scale_slider = Slider(axfreq, 'Scale', 0.1, 20, valinit=scale, valstep=0.1)
scale_slider.on_changed(update_scale)

axfreq = plt.axes([0.22, 0.11, 0.59, 0.03], facecolor='lightgoldenrodyellow')
angle_slider = Slider(axfreq, 'Angle', -math.pi, math.pi, valinit=angle, valstep=0.1)
angle_slider.on_changed(update_angle)

axfreq = plt.axes([0.22, 0.07, 0.59, 0.03], facecolor='lightgoldenrodyellow')
centx_slider = Slider(axfreq, 'Center X', 0, 1, valinit=center_x, valstep=0.05)
centx_slider.on_changed(update_centerx)

axfreq = plt.axes([0.22, 0.03, 0.59, 0.03], facecolor='lightgoldenrodyellow')
centy_slider = Slider(axfreq, 'Center Y', 0, 1, valinit=center_y, valstep=0.05)
centy_slider.on_changed(update_centery)

plt.show()
