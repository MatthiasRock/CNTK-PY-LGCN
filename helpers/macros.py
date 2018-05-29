from helpers.include import *

def reluCustom(inp):
	return C.element_times(relu(inp), 0.1)
	
def paddedConvBNReLULayer(outMap, k, s):
	return Sequential([
		Convolution((k,k), num_filters = outMap, init = he_normal(scale = 0.141421), pad = True, strides = (s,s), bias = True),
		BatchNormalization(map_rank = 1, init_scale = 1, normalization_time_constant = 0, epsilon = 0.00001, use_cntk_engine = False),
		reluCustom
	])
	
def dilatedConvBNReLULayer(outMap, k, d):
	return Sequential([
		Convolution((k,k), num_filters = outMap, init = he_normal(scale = 0.141421), pad = True, dilation = (d,d), bias = True),
		BatchNormalization(map_rank = 1, init_scale = 1, normalization_time_constant = 0, epsilon = 0.00001, use_cntk_engine = False),
		reluCustom
	])
		
def LastConvLayer(outMap, k, s):
	return Convolution((k,k), num_filters = outMap, init = he_normal(scale = 0.141421), pad = True, strides = (s,s), bias = True)