import torch
import torch.nn as nn

""" Convolution layer architecture dictionary that contains each of the below vgg types in list arrays.
    The integer values are the channel sizes and "M" is the maxpooling layers.
    Each type's conv layers always has blocks consisting of: 3x3 convolution layers, padding=1, stride=1
    Image input has to be 224 x 224 RGB image.
    Refer to the official VGG architecture Paper pdf (https://arxiv.org/abs/1409.1556): Table-1 diagram in Page 3. 
    """
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG_net(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels # since each layer's input channel size remains unchanged in VGG architecture
        
        # call create_conv_layers method to create all the necessary layers for desired VGG type
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])

        # create the fully connected (fc) layers shown at the bottom of Table-1
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # 7 comes from input image size 224/(2^(num of maxpool layers)), or 224/(2**5) = 7
            nn.ReLU(),
            nn.Dropout(p=0.5), # Dropout method is an effective technique for regularization and preventing the co-adaptation of neurons
            nn.Linear(4096, 4096), # coming from the previous fc layer, input size is 4096 and output size is 4096 according to Table-1
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        
    """ Forward method that is a requirement to implement using the Pytorch nn module for training model! 
        It functions the same as __call__ in python and it allows the instance (class that contains this method itself) to be called like a method """
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1) # reshape x to flatten it to the Linear part of the data
        x = self.fcs(x) # forward to the fc layers
        return x

    def create_conv_layers(self, vgg_architecture):
        layers = []
        in_channels = self.in_channels

        # vgg_architecture is the conv layer architecture list defined above, which depends on the VGG type
        for x in vgg_architecture:
            # if x (out_channel) is an integer then it is a conv layer
            if type(x) == int:
                out_channels = x
                # combine all the conv layers together one at a time in the for loop
                layers += [
                    nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        kernel_size = (3, 3), # refers to [kernel_size] x kerner_size] convolution in the row of each block matrix
                        stride = (1, 1), # Stride is a parameter that dictates the movement of the kernel, or filter, across the input data
                        padding = (1, 1),
                    ),
                    nn.BatchNorm2d(x), # BatchNorm2d is used to stabilize the optimization process
                    nn.ReLU(), # Rectified Linear Unit (ReLu) is a non-linear activation function that performs on CNN
                ]
                in_channels = x # update the in_channels to be x for the subsequent layer's input channel
            # else x is a maxpool layer    
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))] # kernel size and stride according to official VGG paper

        return nn.Sequential(*layers) # * here is used to unpack the iterable object "layers"


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = VGG_net(in_channels=3, num_classes=1000).to(device)
#     BATCH_SIZE = 3
#     x = torch.randn(3, 3, 224, 224).to(device)
#     assert model(x).shape == torch.Size([BATCH_SIZE, 1000])
#     print(model(x).shape)