# Implementation of Self-recognition with semantic segmentation networks. 

# Training the network – Train_all.ipynb
	1.	In the beginning of the file define the desired parameters of the model and the training (it is possible to choose a few):

	  •	The architectures and the encoder of the model you want to train:

	      o	2 Architectures: Unet and deeplabv3

	      o	3 Encoders: "Xception", Densenet161", "Resnet101"

	  •	Different types of labels: 

	    o	Delta = 1 

	    o	Delta = 3

	  •	Number of output channels: 

	    o	1 

	    o	2
	  •	Loss function: 

	    o	MSE

	    o	BCEWithLogits

	  •	Batch size

	2.	Make sure to change path_model and path_graphs in to the where you want the models and the graphs to be saved. The other paths should indicate where the data is located. 
	3.	Run all cells

# Testing the network – Test_model.ipynb
	1.	Make sure the paths are directed to the folder where the graphs and models are located (should be the same paths as before if you run the testing after the training).
	2.	Define test_new parameter as 1 if you want to calculate the measures on a new dataset 
	3.	Define show_sample_images as 1 if you want to present some samples of segmentations performed on the new dataset
	4.	Run all cells

# Helper Functions:
Scores.py: include the evaluation metrics: IoU and Dice scores.
Dataset_builder:  create one dataset of images and labels from given folders
CreateDIffImages: creating the labels according the choice of the difference between frames.
VideoToImages: converting a video into Images 
