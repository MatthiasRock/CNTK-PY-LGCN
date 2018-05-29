import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings

# Plots the training error and validation error during training
def plot_loss(queue):

	warnings.filterwarnings("ignore", category = matplotlib.cbook.mplDeprecation)	# Suppress warning

	plt.ion()
	plt.figure()
	plt.title('Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	
	while True:
		data = queue.get(False)
		line_training, 		= plt.loglog(data[0], data[1], "r")
		line_validation, 	= plt.loglog(data[0], data[2], "b")
		plt.legend((line_training, line_validation), ('Training', 'Validation'))
		plt.show()
		plt.pause(0.001)
		
		while queue.empty():
			plt.pause(1)