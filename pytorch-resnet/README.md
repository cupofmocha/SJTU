[HOW TO RUN]
This note will explain how to successfully run the codes in this project.

Goal:
To train and predict object classification using popular resnet models.

Prerequisites:
-Make sure the "data" folder contains the necessary training images and validation images
-Make sure to use switch between gpu or cpu usages in train.py and predict.py (by default it uses cpu)
-Install any library dependencies with pip or pip3 install

Execution:
This project supports two different scenarios that you can train, one using a pre-trained pytorch model, and another using own implementation of such model. You can change the batch_size global variable in train.py (default=4).


1st SCENARIO (using pretrained model):
1. To train the model, simply run the command "python train.py --use-pretrained --epochs [x]" in windows cmd or powershell (for example) within the project directory.

 - [x] is the # of epochs to be executed for training

2. After training is completed, you should get a list of .pth model files corresponding to each epoch done under "checkpoints" folder. The .pth file having the LARGEST index number should be used for generating the prediction feature map image.

3. To visualize the predicted classifications as plots (they will also be saved as .jpg files in "saved" folder), run the command "python visualize.py --use-pretrained --model ./checkpoints/checkpoint_epoch[z].pth --viz [i]".

 - [y] = highest epoch index number in the "checkpoints" folder
 - [i] = how many images you want to check/display, then they will be saved in "saved" folder



2nd SCENARIO (using own implementation model):
1. To train the model, simply run the command "python train.py --epochs [x]" in windows cmd or powershell (for example) within the project directory.

 - [x] is the # of epochs to be executed for training

2. After training is completed, you should get a list of .pth model files corresponding to each epoch done under "checkpoints" folder. The .pth file having the LARGEST index number should be used for generating the prediction feature map image.

3. To visualize the predicted classifications as plots (they will also be saved as .jpg files in "saved" folder), run the command "python visualize.py --model ./checkpoints/checkpoint_epoch[z].pth --viz [i]".

 - [y] = highest epoch index number in the "checkpoints" folder
 - [i] = how many images you want to check/display, then they will be saved in "saved" folder