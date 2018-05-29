from helpers.include import *
from helpers.macros import *

#######################
#       Inputs        #
#######################

# Images
features 		= C.input_variable((ImageC, ImageH, ImageW), dtype = np.float32, name = 'features')	# [ImageC x ImageH x ImageW]
featureScaled	= C.element_times(features, 1/128)
featureScaled 	= C.minus(featureScaled, 1)
featureScaled	= C.element_times(featureScaled, 0.1)

# Mask
mask 			= C.input_variable(numLabels, dtype = np.float32)									# [numLabels]
maskX 			= C.reshape(mask, (numLabels, 1, 1))												# [numLabels x 1 x 1]

label 			= {}

# Heatmaps
for x in range(1, numLabels + 1):
	label[x] = C.input_variable((1, ImageH, ImageW), dtype = np.float32)							# [1 x ImageH x ImageW]

#######################
#     Load Kernel     #
#######################
def load_kernel():
	kernelSize 		= 45
	f 				= open(Dir_PathAbs + '/helpers/kernel45.txt')
	kernel45 		= np.fromfile(f, dtype = np.float32, count = -1, sep = " ")						# Read whole *.txt-file
	kernel45 		= np.reshape(kernel45, (1, 1, kernelSize, kernelSize))							# [1 x 1 x 45 x 45]
	f.close()
	kernel_stacked 	= np.tile(kernel45, (numLabels, 1, 1, 1))										# [numLabels x 1 x 45 x 45]
	kernel 			= C.constant(value = kernel_stacked, dtype = np.float32, name = 'kernel')		# [numLabels x 1 x 45 x 45]

	return kernel
		
#######################
#    Create Model     #
#######################	
def create_model(nFilters):

	kernel = load_kernel()																			# [numLabels x 1 x 45 x 45]

	conv15d = Sequential([																			# [nFilters x ImageH x ImageW]
		For(range(5), lambda i: [
			paddedConvBNReLULayer(nFilters, 5, 1),
			Dropout(dropoutRate)
		]),
		For(range(10), lambda i: [
			paddedConvBNReLULayer(nFilters, 3, 1),
			Dropout(dropoutRate)
		])
	])(featureScaled)

	top_main = LastConvLayer(numLabels, 1, 1)(conv15d)												# [numLabels x ImageH x ImageW]
	top_side = paddedConvBNReLULayer(numLabels, 1, 1)(conv15d)										# [numLabels x ImageH x ImageW]

	o1 		= C.convolution(kernel, top_main, groups = numLabels)									# [numLabels x ImageH x ImageW]
	o1		= C.element_times(o1, 0.01, name = 'o1')

	o_side 	= C.convolution(kernel, top_side, groups = numLabels)									# [numLabels x ImageH x ImageW]
	o_side	= C.element_times(o_side, 0.01)

	top_stacked = C.splice(o1, o_side, axis = 0)													# [2*numLabels x ImageH x ImageW]

	bottom = Sequential([																			# [nFilters x ImageH x ImageW]
		For(range(7), lambda i: [
			dilatedConvBNReLULayer(nFilters, 3, 4),
			Dropout(dropoutRate)
		])
	])(top_stacked)

	o2 = C.plus(LastConvLayer(numLabels, 1, 1)(bottom), o1, name = 'o2')							# [numLabels x ImageH x ImageW]

	return o2

#######################
# Criterion function  #
#######################
def create_criterion_function(model):

	label_list = []
	
	# Labels
	for x in range(1, numLabels + 1):
		label_list.append(label[x])

	labels 		= C.splice(*label_list, axis = 0)													# [numLabels x ImageH x ImageW]
	labels 		= C.element_times(labels, 1/255) 

	lconv		= C.convolution(model.find_by_name('kernel'), labels, groups = numLabels) 
	
	nonempty 	= C.reduce_sum(C.reduce_sum(labels, axis = 2), axis = 1)							# [numLabels x 1 x 1]
	weights 	= C.element_times(C.reduce_mean(nonempty), 0.5)
	weights 	= C.plus(0.1, weights, nonempty)													# [numLabels x 1 x 1]
	
	diff_conv1 	= C.minus(model.find_by_name('o1'), lconv)											# [numLabels x ImageH x ImageW]
	diff_conv2 	= C.minus(model, lconv)																# [numLabels x ImageH x ImageW]

	sqErr 		= C.plus(C.element_times(diff_conv1, diff_conv1), C.element_times(diff_conv2, diff_conv2))	# [numLabels x ImageH x ImageW]
	sqErr 		= C.reduce_sum(C.element_times(sqErr, weights, maskX))
	sqErr 		= C.element_times(sqErr, 0.05)
	
	criterion 	= C.combine([sqErr, sqErr])															# (loss, metric)
	
	return criterion