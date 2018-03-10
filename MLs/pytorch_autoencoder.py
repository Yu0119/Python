"""
 Pytorch Autoencoder
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import os
import six
import argparse

# Convert to image arrays
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# Autoencoder Model
class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 64),
            nn.Tanh(),
            nn.Linear(64, 8),
            nn.Tanh(),
            nn.Linear(8, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 64),
            nn.Tanh(),
            nn.Linear(64, 784)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded
        

if __name__ == '__main__':

    # Hyper parameters
    num_epochs = 30
    input_size = 28*28
    learning_rate = 0.001
    use_cuda = False
    
    # Datasets
    images = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    train_loader = torch.utils.data.DataLoader(
        images,
        shuffle=True,
        batch_size=batch_size
    )

    # Create model
    autoencoder = AutoEncoder()
    
    if use_cuda:
        autoencoder.cuda()
      
    print(autoencoder)

    import torch.optim as optim

    # Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Create output directory
    if not os.path.exists('output'): os.mkdir('output')

    # Train    
    for epoch in six.moves.range(num_epochs):

        sum_loss = 0.0
        for i, (images, _) in enumerate(train_loader):
            dtype = torch.FloatTensor
            # datasets
            images = images.view(-1, 28*28).type(dtype)
            
            if use_cuda:
                images = images.cuda()
              
            images = Variable(images)
            
            optimizer.zero_grad()

            # forward & backprop
            _, encoded = autoencoder(images)
            loss = criterion(encoded, images)
            loss.backward()
            optimizer.step()

            sum_loss += loss.data[0]

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}], loss: {}'.format(epoch+1, num_epochs, i+1, sum_loss))

                # Initialize sum loss
                sum_loss = 0.0

        pic = to_img(encoded.cpu().data)
        save_image(pic, './output/image_epoch{}.png'.format(epoch))
