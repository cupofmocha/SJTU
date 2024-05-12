[HOW TO RUN]
This note will explain how to successfully run the codes in this project.

Prerequisites:
-Make sure the "Train" folder contains the necessary images and masks
-Make sure to use switch between gpu or cpu usages in train.py and predict.py (by default it uses cpu)
-Put images and masks under the folder "Train", and make sure to set the variable num_train appropriately in "train.py" (not exceeding total number of images but also not too small) for the training to work
-Install any library dependencies with pip or pip3 install

Execution:
1. To train the model, simply run the command "python train.py --epochs [x] --batch-size [y]" in windows cmd or powershell (for example) within the project directory, where [x] is the # of epochs to be executed for training, [y] is the batch size of images for training.

2. After training is completed, you should get a list of .pth model files corresponding to each epoch done under "checkpoints" folder. The .pth file having the LARGEST index number should be used for generating the prediction feature map image.

3. To generate the desired prediction image using an original image, run the command "python predict.py --model ./checkpoints/checkpoint_epoch[z].pth -i [A] --viz --output [B]"

This is a bit complicated but you need to specify 3 things:
 - [z] = highest epoch index number in the "checkpoints" folder
 - [A] = input image directory (original image's directory), for example, "./Train/Training_Images/ISIC_0000000.jpg"
 - [B] = output segmentation (feature map) file name that MUST end with ".jpg", you can name it however you want and it will be saved under "asset" folder

After these 3 steps you will obtain a blended image (./asset/blended_image.jpg) that can be used to visualize the UNET training feature map results for any specific image. 