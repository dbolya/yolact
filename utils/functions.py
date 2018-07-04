

class MovingAverage():
	""" Keeps an average window of the specified number of items. """

	def __init__(self, max_window_size=100):
		self.max_window_size = max_window_size
		self.window = []
		self.sum = 0

	def add(self, elem):
		""" Adds an element to the window, removing the earliest element if necessary. """
		self.window.append(elem)
		self.sum += elem

		if len(self.window) > self.max_window_size:
			self.sum -= self.window.pop(0)

	def get_avg(self):
		""" Returns the average of the elements in the window. """
		return self.sum / max(len(self.window), 1)

	def __str__(self):
		return str(self.get_avg())
	
	def __repr__(self):
		return repr(self.get_avg())

