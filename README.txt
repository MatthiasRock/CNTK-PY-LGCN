Fully-Convolutional Local-Global Context Network for Robust Facial Landmark Detection

### General Information

This is the Python version of the original BrainScript network: http://www.mmk.ei.tum.de/cvpr2018/

Details of the network can be found in:

Daniel Merget, Matthias Rock, Gerhard Rigoll: "Robust Facial Landmark Detection via a Fully-Convolutional Local-Global Context Network". In: Proceedings of the International Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2018.


### Setup Information

CNTK Development Setup: tested with version 2.3 and 2.4 (script-driven installation)
	- We recommend using CNTK 2.3 as the version 2.4 is much slower for our network

	
### Getting Started

If you want to start training or testing with the batch files "start_train.bat" and "start_prediction.bat", you first have to execute the script "use_development_setup.bat" once.
Maybe the paths inside this batch file have to be adapted.

Note: "QuickEdit" mode should be disabled in command line when running our CNTK python scripts.


### Training

1) Create the following *.txt files in "./reader_files/train/" containing the paths of the images/heatmaps with tab separated zeros and create the nanlist:
	- img.txt
	- lm1.txt
	- ...
	- lm68.txt
	- nanlist.txt
	The nanlist contains a mask for all landmarks. Typically, we just set all entries to one. In this case all lines look as follows:
	|mask 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
2) Do the same for the validation set in "./reader_files/val/"
3) Adapt the parameters in "train.py"
4) Run "train.py" in your Python environment or just start the training with the batch file "start_train.bat"

The logfiles and the models will be stored in "./output/".


### Test

1) Run the training first
2) Create the file "img.txt" in "./reader_files/predict/" containing the paths of the images with tab separated zeros
3) Adapt the parameters in "predict.py"
4) Run "predict.py" in your Python environment or just start the prediction with the batch file "start_prediction.bat"

The predicted heatmaps will be stored in "./output/prediction.mat".