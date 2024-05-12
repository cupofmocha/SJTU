import torch
import torch.nn as nn

# Basic block structure class that is used by the resnet architecture (each block is a matrix), from the official thesis papers
class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride = 1
    ):
        super(block, self).__init__() # super() enables method overriding and inheritance of classes
        self.expansion = 4 # output channel size of each layer is 4 times that of the input
        
        """ 1st convolution layer of a block, i.e. first row in the 3 x 2 matrix. 
        Refer to the official ResNet architecture Paper pdf (https://arxiv.org/abs/1512.03385): Table-1 diagram in Page 5. """
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size = 1, # refers to [kernel_size] x kerner_size] convolution in the row of each block matrix
            stride = 1, # Stride is a parameter that dictates the movement of the kernel, or filter, across the input data
            padding = 0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels) # BatchNorm2d is used to stabilize the optimization process
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size = 3,
            stride=stride,
            padding = 1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion, # output channel size of each layer is 4 times that of the input
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU() # Rectified Linear Unit (ReLu) is a non-linear activation function that performs on CNN
        self.identity_downsample = identity_downsample # identity downsample is to ensure the later layer's identity mapping  have the same shape
        self.stride = stride

    """ Forward method that is a requirement to implement using the Pytorch nn module for training model! 
        It functions the same as __call__ in python and it allows the instance (class that contains this method itself) to be called like a method """
    def forward(self, x):
        identity = x.clone()

        # chain the layers together in each block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        # if identity mapping's shape needs to be changed in some way use identity downsample
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # finally add identity to x and assign ReLu to x to perform on it
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers_list, image_channels, num_classes): 
        """ -- layers_list is a list that contains how many times each block per layer is being reused
               for example, for ResNet50, layers is [3, 4, 6, 3] referring to layer conv2_x, conv3_x, conv4_x, and conv5_x.
          -- image_channels (=3 for RGB input images for example)
          -- num_classes refers to our data, i.e. how many different classifications/categories """
        super(ResNet, self).__init__()
        self.in_channels = 64 # 1st layer, conv1, accepts 64 input channels
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        # Make each layer accordingly
        self.layer1 = self._make_layer(
            block, layers_list[0], intermediate_channels = 64, stride = 1
        )
        self.layer2 = self._make_layer(
            block, layers_list[1], intermediate_channels = 128, stride= 2
        )
        self.layer3 = self._make_layer(
            block, layers_list[2], intermediate_channels = 256, stride = 2
        )
        self.layer4 = self._make_layer(
            block, layers_list[3], intermediate_channels = 512, stride = 2
        )

        ### Adaptive Average pooling: performs summarization over convolutional feature maps, while capturing the essential behavior of the feature map itself ###
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # (1, 1) is the output size (Height x Width) in this case for an image
        # fully connected (fc) layer at the end
        self.fc = nn.Linear(512 * 4, num_classes)

    # forward method
    """ Construct the entire ResNet architecture below sequentially by matching inputs and outputs in CORRECT order, 
        these layers are the convolution layers for the entire network (different than layers(rows) in each block)
        i.e. conv2_x, conv3_x, conv4_x, and conv5_x, again refer to the official ResNet architecture """
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # reshape so it can be sent to the fc layer
        x = self.fc(x)

        return x

    # make layer method to construct each of the conv layers defined by the official ResNet architecture
    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        # num_residual_blocks refers to the number of times the block is being reused
        identity_downsample = None
        layers = []

        """ In scenarios where we halve the input channel size e.g., 256x256 -> 128x128 (stride=2) as we go down each block, or number of channels changes,
            we need to adapt or modify the Identity (skip connection) to be the same shape so it will be able to be added to the layers ahead.
            The halving happens during the first layer of each residual block as it accepts input channels from the previous block's output channels """
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size = 1,
                    stride = stride,
                    bias = False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        ### below residual block layer is responsible for changing the number of output channels for a layer! ###
        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size multiplier is always 4 for ResNet 50, ResNet 101, ResNet 152 as we go down the layer
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        
        ### connect each layer's own intermediate blocks together via appending sequentially to form the layer ###
        for i in range(num_residual_blocks - 1): # -1 since the residual block above that changes output channels is already accounted for
            layers.append(block(self.in_channels, intermediate_channels)) # e.g. a process can be input 256 -> 64, followed by 64*4 (turns back into 256 again) in the intermediate output

        return nn.Sequential(*layers) # * here is used to unpack the iterable object "layers"


def ResNet50(num_classes, img_channel=3):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(num_classes, img_channel=3):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(num_classes, img_channel=3):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


# def test():
#     BATCH_SIZE = 4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = ResNet101(img_channel=3, num_classes=1000).to(device)
#     y = net(torch.randn(BATCH_SIZE, 3, 224, 224)).to(device)
#     assert y.size() == torch.Size([BATCH_SIZE, 1000])
#     print(y.size())


# if __name__ == "__main__":
#     test()