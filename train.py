from helpers.plot import *

######################################
#     Main program for training      #
######################################
if __name__ == '__main__':
	from helpers.include import *
	from helpers.network import *

	#############################
	#   Adapt these parameters  #
	#############################
	Epochs 				= 100
	minibatchSize_Train	= 16
	minibatchSize_Val 	= 16
	nFilters 			= 128
	learning_rate 		= [
		#	Epoch nr.	LearningRate
		(	1, 			0.001		),
		(	11, 		0.0005		),
		(	21, 		0.0002		),
		(	31, 		0.0001		),
		(	41, 		0.00005		),
		(	51, 		0.00002		)
	]

	#############################
	#  Filenames & Directories  #
	#############################
	TrainDir_PathAbs		= Dir_PathAbs 	   	+ '/reader_files/train'
	FeatureTrain_PathAbs	= TrainDir_PathAbs 	+ '/img.txt'
	LabelTrain_PathAbs		= TrainDir_PathAbs 	+ '/lm%d.txt'
	MaskTrain_PathAbs		= TrainDir_PathAbs 	+ '/nanlist.txt'

	ValDir_PathAbs			= Dir_PathAbs 	 	+ '/reader_files/val'
	FeatureVal_PathAbs 		= ValDir_PathAbs 	+ '/img.txt'
	LabelVal_PathAbs		= ValDir_PathAbs 	+ '/lm%d.txt'
	MaskVal_PathAbs			= ValDir_PathAbs 	+ '/nanlist.txt'

	TrainLogFileName		= 'train_val%s.log'

	# Determine the logfile path
	LogFile_PathAbs = LogFileDir_PathAbs + '/' + TrainLogFileName % ""
	i = 1

	# Do not overwrite the existing logfile
	while os.path.isfile(LogFile_PathAbs):
		i += 1
		LogFile_PathAbs = LogFileDir_PathAbs + '/' + TrainLogFileName % str(i)

	# Create logfile directory
	if not os.path.exists(LogFileDir_PathAbs):
		os.makedirs(LogFileDir_PathAbs)

	#############################
	#   Determine epoch sizes   #
	#############################
	
	epochSize_Train = 0
	epochSize_Val 	= 0

	# Epoch size of training set
	with open(FeatureTrain_PathAbs) as file:
		for line in file:
			if line.strip():
				epochSize_Train += 1
	
	# Epoch size of validation set
	with open(FeatureVal_PathAbs) as file:
		for line in file:
			if line.strip():
				epochSize_Val += 1

	#############################
	#    Create / load model    #
	#############################
	
	# Determine the last stored model
	LastModel_PathAbs = False
	i = 0
	while os.path.isfile(Models_PathAbs % i):
		LastModel_PathAbs = Models_PathAbs % i
		i += 1

	epoch_start = max(1, i)

	# Load last model if one already exists
	if LastModel_PathAbs:
		model		= load_model(LastModel_PathAbs)
		features	= model.find_by_name('features')
	# Create model
	else:
		model = create_model(nFilters)
	
	# Create criterion function
	criterion = create_criterion_function(model)

	#############################
	#      Read input data      #
	#############################

	# Deserializer (train)
	deserializerTrain_list = [create_feature_deserializer(FeatureTrain_PathAbs)]
	deserializerTrain_list.extend(create_label_deserializer_list(LabelTrain_PathAbs))
	deserializerTrain_list.append(create_mask_deserializer(MaskTrain_PathAbs))

	readerTrain = MinibatchSource(deserializerTrain_list, trace_level = 1, randomize = True)

	# Define mapping from reader streams to network inputs (train)
	input_mapTrain 				= {}
	input_mapTrain[features] 	= readerTrain.streams.features
	input_mapTrain[mask]		= readerTrain.streams.mask

	for x in range(1, numLabels + 1):
		exec("input_mapTrain[label[%d]] = readerTrain.streams.label%d" % (x, x))

	# Deserializer (val)
	deserializerVal_list = [create_feature_deserializer(FeatureVal_PathAbs)]
	deserializerVal_list.extend(create_label_deserializer_list(LabelVal_PathAbs))
	deserializerVal_list.append(create_mask_deserializer(MaskVal_PathAbs))

	readerVal = MinibatchSource(deserializerVal_list, trace_level = 1, randomize = True)

	# Define mapping from reader streams to network inputs (val)
	input_mapVal			= {}
	input_mapVal[features] 	= readerVal.streams.features
	input_mapVal[mask]		= readerVal.streams.mask

	for x in range(1, numLabels + 1):
		exec("input_mapVal[label[%d]] = readerVal.streams.label%d" % (x, x))

	#############################
	#    Learning parameters    #
	#############################

	lr_list_length = len(learning_rate)

	# Adapt learningrate to the epoch number of the loaded model
	for x in range(lr_list_length):
		learning_rate[x] = (max(1, learning_rate[x][0] - epoch_start + 1), learning_rate[x][1])

	learning_rate.append((learning_rate[lr_list_length-1][0] + 1, learning_rate[lr_list_length-1][1]))
	lr_list = []

	# Transform the list to the necessary data format
	for x in range(1, lr_list_length+1):
		epoch_length = learning_rate[x][0] - learning_rate[x-1][0]
		
		if epoch_length > 0:
			lr_list.append((epoch_length, learning_rate[x-1][1]))

	lr_schedule = learning_parameter_schedule_per_sample(lr_list, epoch_size = epochSize_Train)
	m_schedule 	= momentum_schedule(0.9)
	learner 	= momentum_sgd(model.parameters, lr = lr_schedule, momentum = m_schedule, gradient_clipping_threshold_per_sample = 0.1, gradient_clipping_with_truncation = True, epoch_size = epochSize_Train)

	#############################
	#         Training          #
	#############################

	progress_printer	= ProgressPrinter(tag = 'Training', freq = 50, log_to_file = LogFile_PathAbs, num_epochs = Epochs, metric_is_pct = False)
	trainer 			= C.Trainer(model, criterion, learner, progress_writers = progress_printer)
	
	plot_threshold		= epoch_start + 1
	plotdata 			= defaultdict(list)
	curr_samples_per_s	= 0;
	
	print("##################################################")
	print("################# Start Training #################")
	print("##################################################\n")
	
	# If a model already exists
	if LastModel_PathAbs:
		print("# There are already existing models in the output folder!")
		print("# In case you want to start a new training, delete the output folder and restart the script!\n")
		print("# The last training is continued...\n")
	# If there does not exist a trained model yet
	else:
		model.save(Models_PathAbs % 0)	# Save initial model

	# Perform model training
	for epoch in range(epoch_start, Epochs + 1):
		progress_printer.log("Epoch[%d of %d]" % (epoch, Epochs))
		sample_count 	= 0
		train_error 	= 0
		t_start 		= time.time()
		
		print("Epoch %d of %d:" % (epoch, Epochs))
		print("\tTraining:\t   0 %% (% 5.1f samples/s)"% curr_samples_per_s, end = '', flush = True)
		
		# Train for all minibatches
		while sample_count < epochSize_Train:
			t_start_mb		= time.time()
			currentMBsize 	= min(minibatchSize_Train, epochSize_Train - sample_count)
			data 			= readerTrain.next_minibatch(currentMBsize, input_map = input_mapTrain)
			
			trainer.train_minibatch(data)
			
			sample_count 	+= currentMBsize
			train_error 	+= currentMBsize*trainer.previous_minibatch_loss_average
			
			sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
			curr_samples_per_s = currentMBsize/(time.time()-t_start_mb)
			print("% 3d %% (% 5.1f samples/s)" % (math.floor(100*sample_count/epochSize_Train), curr_samples_per_s), end = '', flush = True)

		model.save(Models_PathAbs % epoch)
		trainer.summarize_training_progress()
		train_error = train_error/sample_count
		sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
		print(" (%5.1f samples/s)" % (epochSize_Train/(time.time() - t_start)))
		
		# Validation:
		print("\tValidation:\t   0 %", end = '', flush = True)
		sample_count 	= 0
		val_error 		= 0
		t_start 		= time.time()
		
		# Validate for all minibatches
		while sample_count < epochSize_Val:
			currentMBsize 	= min(minibatchSize_Val, epochSize_Val - sample_count)
			data 			= readerVal.next_minibatch(currentMBsize, input_map = input_mapVal)
			
			sample_count 	+= currentMBsize
			val_error 		+= currentMBsize*trainer.test_minibatch(data)
			
			sys.stdout.write('\b\b\b\b\b')
			print("% 3d %%" % (math.floor(100*sample_count/epochSize_Val)), end = '', flush = True)
			
		trainer.summarize_test_progress()
		val_error = val_error/sample_count
		print(" (%5.1f samples/s)" % (epochSize_Val/(time.time() - t_start)))
		
		#############################
		#           Plot            #
		#############################
		plotdata["epoch"].append(epoch)
		plotdata["train_error"].append(train_error)
		plotdata["val_error"].append(val_error)
			
		if epoch == plot_threshold:
			queue = multiprocessing.Queue()
			p = multiprocessing.Process(target = plot_loss, args = (queue,))
			p.start()
		if epoch >= plot_threshold:
			queue.put((plotdata['epoch'], plotdata["train_error"], plotdata["val_error"]))
		
	print("\nTraining Finished!")