import torch
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary

import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Global variables
data_dir = './train'
test_dir = './test'
dir_checkpoint = Path('./checkpoints/')

def get_args():
    parser = argparse.ArgumentParser(description='Predict object classifications after performing training')
    parser.add_argument('--use-pretrained', '-u', action="store_true", help='Use Pretrained network')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=3, help='Number of epochs to run')  
    return parser.parse_args()


NUM_WORKERS = os.cpu_count()

# Function for creating dataloaders
def create_dataloaders(
    data_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):

    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(data_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names from train dataset
    class_names = train_data.classes
    
    #print(class_names)

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

def set_seeds(seed: int=42):
        """Sets random sets for torch operations.

        Args:
            seed (int, optional): Random seed to set. Defaults to 42.
        """
        # Set the seed for general torch operations
        torch.manual_seed(seed)
        # Set the seed for CUDA torch operations (ones that happen on the GPU)
        torch.cuda.manual_seed(seed)
# Create image size
IMG_SIZE = 224

# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])           
#print(f"Manually created transforms: {manual_transforms}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu') # use this for cpu


# Set the batch size
BATCH_SIZE = 32 

# Create data loaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    data_dir=data_dir,
    test_dir=test_dir,
    transform=manual_transforms, 
    batch_size=BATCH_SIZE
)

train_dataloader, test_dataloader, class_names

#region Input Image Pre-processing
# Step 1: This is the "Linear Projection of Flattened Patches" step from (An Image is worth 16x16 words Paper)
# Tasks:
# a. turn an image into 16*16 patches, or (224/16)*(224/16) = 196 patches per image

# b. flatten the patch feature maps into a single dimension

# c. Convert the output into Desired Output (flattened 2D patches): (196, 768) -> N×(P2⋅C) #Current shape: (1, 768, 196)

# Converts 2D input image into a 1D sequence learnable embedding vector (patches)
class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels:int=3, # color channels, rgb
                 patch_size:int=16,
                 embedding_dim:int=768 # comes from 16*16*in_channels, so each patch after flattened has 768 different values
                 ):
        super().__init__()
        self.patch_size = patch_size

        # ***layer to turn image into patches (using 1 Convolution block, can think of it as "1 Conv layer in a ResNet block")***
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        
        # layer to flatten the patches into 1D
        self.flatten = nn.Flatten(start_dim=2,end_dim=3)

    # Forward method to convert output into Desired Output
    def forward(self, x):
        # Create assertion to ensure inputs are of the correct shape
        img_resolution = x.shape[-1]
        assert img_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {img_resolution}, patch size: {self.patch_size}"

        # forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        # Make sure the output shape has the right order 
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
#endregion

# Make Multi-head Self Attention (MSA) Block
# The MSA block is responsible for computing pairwise similarity between elements in the array
class MultiheadSelfAttentionBlock(nn.Module): 
    # Initialize the class with hyperparameters from Table 1 in the Paper
    def __init__(self, 
                 embedding_dim:int=768, # hidden size D for ViT Base model in Table 1
                 num_heads:int=12, # heads for ViT Base model
                 attn_dropout:float=0
                ):
        super().__init__()

        # Create the Norm Layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the MSA layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
        
    # create forward method to pass data between layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, #Q,K,V needed for every 1D patch
                                             key=x,
                                             value=x,
                                             need_weights=False)
        return attn_output


# Make MLP Block
# The MLP is the artificial neural network for deep-learning training
class MLPBlock(nn.Module):
    # Initialize the class with hyperparameters from Table 1 & 3 in the Paper
    def __init__(self, 
                 embedding_dim:int=768, # hidden size D for ViT-Base in Table 1
                 mlp_size:int=3072, # MLP size for ViT-Base
                 dropout:float=0.1 # dropout value for ViT-Base in Table 3
                ):
        super().__init__()

        # Create the Norm Layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the Multilayer Perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
        
    # create forward method to pass data between layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
    
# Create the Transformer Encoder 
# By combining the normalized MSA and MLP blocks, we can create the Transformer Encoder layer according to Figure 1 of the Paper
class TransformerEncoderBlock(nn.Module):
    # Initialize the class with hyperparameters from Table 1 & 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout:float=0):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    # Create forward method to link all the blocks together
    def forward(self, x):
        x += self.msa_block(x)
        x += self.mlp_block(x)
        return x

#region Create the final Visition Transformer (ViT-Base)
# Combine the previous blocks together with the embeddings
class ViT(nn.Module):
    # Initialize the class with hyperparameters from Table 1 & 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16,
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers 
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__()

        # Check that the input image size is divisible by the patch size 
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
        
        # Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2
                 
        # Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)
        
        # Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim), requires_grad=True)
                
        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

#endregion



if __name__ == '__main__':
    ## Visualize an image to check whether dataloader is properly working
    # Get a batch of images
    image_batch, label_batch = next(iter(train_dataloader))

    # Get a single image from the batch
    image, label = image_batch[0], label_batch[0]

    # View the batch shapes
    print(image.shape, label)
    # Plot image with matplotlib
    # plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    # plt.title(class_names[label])
    # plt.axis(False)
    # plt.show()

    #region Test PatchEmbedding layer
    # patch_size =16

    # set_seeds()

    # # Create an instance of patch embedding layer
    # patchify = PatchEmbedding(in_channels=3,
    #                         patch_size=16,
    #                         embedding_dim=768)

    # # Pass a single image through
    # print(f"Input image shape: {image.unsqueeze(0).shape}")
    # patch_embedded_image = patchify(image.unsqueeze(0)) # add an extra batch dimension on the 0th index, otherwise will error
    # print(f"Output patch embedding shape: {patch_embedded_image.shape}")
    #endregion

    #region Patch & Position Embedding
    # Step 2: This is the "Patch + Position Embedding" step from the Paper
    # Now add the the learnable class embedding and position embeddings
    # From start to positional encoding: All in 1 cell

    set_seeds()

    # 1. Set patch size
    patch_size = 16

    # 2. Print shape of original image tensor and get the image dimensions
    print(f"Image tensor shape: {image.shape}")
    height, width = image.shape[1], image.shape[2]

    # 3. Get image tensor and add batch dimension
    x = image.unsqueeze(0)
    print(f"Input image with batch dimension shape: {x.shape}")

    # 4. Create patch embedding layer
    patch_embedding_layer = PatchEmbedding(in_channels=3,
                                        patch_size=patch_size,
                                        embedding_dim=768)

    # 5. Pass image through patch embedding layer
    patch_embedding = patch_embedding_layer(x)
    print(f"Patching embedding shape: {patch_embedding.shape}")

    # 6. Create class token embedding
    batch_size = patch_embedding.shape[0]
    embedding_dimension = patch_embedding.shape[-1] # 768
    class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                            requires_grad=True) # make sure it's learnable
    print(f"Class token embedding shape: {class_token.shape}")

    # 7. Pre-pend class token embedding to patch embedding
    patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
    print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

    # 8. Create position embedding
    number_of_patches = int((height * width) / patch_size**2)
    position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension),
                                    requires_grad=True) # make sure it's learnable

    # 9. Add position embedding to patch embedding with class token
    patch_and_position_embedding = patch_embedding_class_token + position_embedding
    print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")
    #patch_and_position_embedding

    print(patch_embedding_class_token)  #1 is added in the beginning of each
    #endregion

    transformer_encoder_block = TransformerEncoderBlock()
    