from helpers.include import *

#############################
#   Adapt these parameters  #
#############################
minibatchSize = 16

#############################
#  Filenames & Directories  #
#############################
TestDir_PathAbs		= Dir_PathAbs 	   	+ '/reader_files/predict'
FeatureTest_PathAbs	= TestDir_PathAbs 	+ '/img.txt'
OutputFile_PathAbs	= OutputDir_PathAbs + '/prediction.mat'

##############################
# Determine number of images #
##############################
number_images = 0

with open(FeatureTest_PathAbs) as file:
	for line in file:
		if line.strip():
			number_images += 1		

####################################
# Extract best model from logfiles #
####################################
log_content = ""

# Read all logfiles
for file in os.listdir(LogFileDir_PathAbs):
	if file.endswith(".log"):
		# Read content of the logfile
		f 			= open(os.path.join(LogFileDir_PathAbs, file),'r')
		log_content	= log_content + f.read()
		f.close()

# Extract the validation errors of all epochs
result = re.findall('Finished Evaluation \[(\d+)\]: Minibatch\[\d+\-\d+\]: metric = (\d+.\d+)', log_content, flags=re.DOTALL)

# Find out the best epoch
best_model = min(result, key=lambda x: float(x[1]))
best_model = int(best_model[0])
			
#############################
#        Load model         #
#############################

model		= load_model(Models_PathAbs % best_model)
features	= model.find_by_name('features')

#############################
#      Read input data      #
#############################

reader = MinibatchSource(create_feature_deserializer(FeatureTest_PathAbs), trace_level = 1, randomize = False)

# Define mapping from reader stream to network input
input_map 				= {}
input_map[features] 	= reader.streams.features

#############################
#        Prediction         #
#############################
sample_count 	= 0
output 			= np.zeros((number_images, numLabels, ImageH, ImageW), dtype = np.float32)

print("##################################################")
print("##############   Start Prediction   ##############")
print("##################################################\n")
print("Using model of epoch %d\n" % best_model)
print("Prediction:   0 %% (% 5.1f samples/s)"% 0, end = '', flush = True)

while sample_count < number_images:
	t_start_mb		= time.time()
	currentMBsize 	= min(minibatchSize, number_images - sample_count)
	data 			= reader.next_minibatch(currentMBsize, input_map = input_map)
	output_mb 		= model.eval(data)
	
	output[sample_count:sample_count+currentMBsize,] = np.squeeze(output_mb)
	sample_count += currentMBsize

	sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
	print("% 3d %% (% 5.1f samples/s)" % (math.floor(100*sample_count/number_images), currentMBsize/(time.time()-t_start_mb)), end = '', flush = True)

sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
print("                  ")

print("\nSaving file...", end = '', flush = True)
sio.savemat(OutputFile_PathAbs, {'pred':np.transpose(output)})
print("Finished!\n")