from __future__ import print_function
import os
import re
import sys
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import multiprocessing
import pydot_ng
import cntk as C
import numpy as np
from cntk.ops import *
from cntk.layers.typing import *
from cntk.losses import squared_error
import time
import scipy.io as sio
from pathlib import Path
from _cntk_py import set_computation_network_trace_level, force_deterministic_algorithms
import cntk.io.transforms as xforms
from cntk.ops.functions import load_model
from cntk.layers import *
from cntk.device import try_set_default_device, gpu, cpu
from cntk.io import MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDef, StreamDefs
from cntk.learners import momentum_sgd, learning_parameter_schedule_per_sample, UnitType, momentum_schedule
from cntk.logging import ProgressPrinter
import cntk.logging.graph as logging
from IPython.core.debugger import Tracer
from cntk.initializer import *

try_set_default_device(gpu(0))
set_computation_network_trace_level(0)

#############################
#   Adapt these parameters  #
#############################
numLabels 			= 68
ImageW 				= 96
ImageH 				= 96
dropoutRate			= 0.1
Grayscale 			= True
OutputDir			= 'output'
ModelsPath 			= 'models/model.%d'
LogFileDir			= 'logs'

#############################
#  Filenames & Directories  #
#############################
Dir_PathAbs 			= os.path.dirname(os.path.dirname(os.path.abspath(__file__)).replace('\\','/'))
OutputDir_PathAbs 		= Dir_PathAbs + '/' + OutputDir
Models_PathAbs			= OutputDir_PathAbs + '/' + ModelsPath
LogFileDir_PathAbs		= OutputDir_PathAbs + '/' + LogFileDir

if Grayscale:
	ImageC = 1
else:
	ImageC = 3

#############################
#      Reader functions     #
#############################

# Reader for images
def create_feature_deserializer(path):
	transforms 		= [xforms.scale(width = ImageW, height = ImageH, channels = ImageC, interpolations = "linear")]
	deserializer 	= ImageDeserializer(
		path,
		StreamDefs(
			features = StreamDef(field = 'image', transforms = transforms),
			ignored	 = StreamDef(field = 'label', shape = 1)
		)
	)
	deserializer['grayscale'] = Grayscale
	
	return deserializer

# Reader for mask
def create_mask_deserializer(path):
	return CTFDeserializer(
		path,
		StreamDefs(
			mask = StreamDef(field = 'mask', shape = numLabels)
		)
	)

# Reader for labels/heatmaps
def create_label_deserializer_list(path):
	transforms 	= [xforms.scale(width = ImageW, height = ImageH, channels = 1, interpolations = "linear")]
	list	 	= []
	
	for x in range(1, numLabels + 1):
		deserializer = ImageDeserializer(
			path % x,
			eval("StreamDefs(label%d = StreamDef(field = 'image', transforms = transforms), ignored%d = StreamDef(field = 'label', shape = 1))" % (x, x))
		)
		deserializer['grayscale'] = True
		list.append(deserializer)
	
	return list