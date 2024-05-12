import argparse
import torch
import matplotlib.pyplot as plt
from train import *

save_dir = './saved'

def get_args():
    parser = argparse.ArgumentParser(description='Visualize object classification results from input images after performing training')
    parser.add_argument('--use-pretrained', '-u', action="store_true", help='Use Pretrained network')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help='Specify the file in which the model is stored')    
    parser.add_argument('--viz', '-v', dest='num_images', metavar='V', type=int, default=0, help='To visualize the number of final prediction images as they are processed')
    return parser.parse_args()

# Function to show image and save it
def show_image(input, saveimage_name, title=None):
    # input data is (imagesize,imagesize,channels), but imshow takes (channels,imagesize,imagesize)
    # hence we need to flip the orders via transpose()
    input = input.numpy().transpose((1, 2, 0))

    ### Common Parameters required to normalize images for imshow() to work in Pytorch, based on project ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.figure(figsize = (4, 4)) # adjust figure size
    if title is not None:
        plt.title(title)
    plt.imshow(input)
    
    plt.pause(0.001)  # pause a bit so that plots are updated correctly
    plt.savefig(os.path.join(save_dir, saveimage_name))
    plt.show(block=True) # use block=True so that the plot will not automatically close on itself


# Function to visualize final training results
def visualize_model(checkpoint_filename, num_images):
    # This is for if using a pretrained resnet18 network (Transfer Learning)
    if args.use_pretrained:
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        # The size of each output sample is set to 2.
        model.fc = nn.Linear(num_ftrs, len(class_names))
    else:
    # Otherwise use my own ResNet implementation (accuracy is lower..)
        model = ResNet152(num_classes, img_channel=3) # 2 categories: bees and ants
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)    
    
    # load a checkpoint file to the model    
    state_dict = torch.load(checkpoint_filename, map_location=device)    
    model.load_state_dict(state_dict)
    
    was_training = model.training
    model.eval()
    image_counter = 0

    if (num_images > 0):
        # Loop until all the required predicted images are visualized
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)

                for i in range(inputs.size()[0]):
                    #print(predictions)
                    image_counter += 1
                    
                    # plot title that tells what the model's prediction class is for each image
                    plot_title = f'predicted: {class_names[predictions[i]]}'

                    # output filename for saving plot image (results)
                    out_filename = "image " + str(image_counter) + ".jpg"
                    show_image(inputs.cpu().data[i], out_filename, plot_title)
                    
                    if image_counter == num_images:
                        model.train(mode = was_training)
                        return
            model.train(mode = was_training)
            
if __name__ == '__main__':
    args = get_args()
    
    if (args.use_pretrained):
        model_type = "pre-trained model"
    else:
        model_type = "own model"
        
    print("Loading " + model_type + " checkpoint file " + args.model)
    visualize_model(args.model, num_images=args.num_images)
    
    