import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc = nn.Sequential( 
            nn.Linear(256*8*8, 4096),
            nn.ReLU(), 
            nn.Linear(4096, 50),
        )

    def forward(self, x):
        out = self.cnn(x)
        output1 = out
        out = out.view(out.size()[0], -1)
        output2 = self.fc(out)
        return output1, output2

class VGG16_FCN32s(nn.Module):
    def __init__(self, n_class=7):
        super(VGG16_FCN32s, self).__init__()
        self.encoder_model = models.vgg16(pretrained=True)
        self.encoder_model = self.encoder_model.features
        self.decoder_model = nn.Sequential(
            # Conv2d ( in_channels, out_channels, kernel_size, stride, padding )
            nn.Conv2d(512 , 4096, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096 , 4096, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096 , n_class, 1),
            nn.ConvTranspose2d(n_class, n_class, 32, 32, bias=False)
        )

    def forward(self, x):
        encoded_data = self.encoder_model(x)
        decoded_data = self.decoder_model(encoded_data)
        return decoded_data
