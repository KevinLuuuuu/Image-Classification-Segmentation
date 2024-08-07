import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset, DataLoader
import os
import torch.nn as nn
import torchvision.models as models
import timm
from dataset import ImageDataset, SegmentationDataset
from model import Net, VGG16_FCN32s
from types import SimpleNamespace
from timm.optim.optim_factory import create_optimizer
from PIL import Image

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" 
print(device)
torch.cuda.empty_cache()

# set random seed
seed = 5203
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 8
dataset_path = "./hw1_data/p2_data/train"
dataset_path2 = "./hw1_data/p2_data/validation"

train_set = SegmentationDataset(dataset_path, transform=train_transform)
valid_set = SegmentationDataset(dataset_path2, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

'''
# model A, mean_iou: 0.535624
model = VGG16_FCN32s().to(device)
args = SimpleNamespace()
args.weight_decay = 0
args.lr = 1e-3
args.opt = 'Adam'
args.momentum = 0.9
optimizer = create_optimizer(args, model)
epochs = 50
'''

# model B
model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.aux_classifier = None
model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, 7).to(device)
args = SimpleNamespace()
args.weight_decay = 0
args.lr = 1e-4
args.opt = 'AdaBelief' #'lookahead_adam' to use `lookahead`
args.momentum = 0.9
optimizer = create_optimizer(args, model)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-3,total_steps=5000, pct_start=0.2)
epochs = 20

criterion = nn.CrossEntropyLoss()
best_loss = 1000

for epoch in range(epochs):

    model.train()
    train_loss = 0
    train_loss_record = []
    train_correct = 0

    for i, (sat_image, mask_image) in enumerate(tqdm(train_loader)):
        sat_image, mask_image = sat_image.to(device), mask_image.to(device)
        optimizer.zero_grad()
        #output = model(sat_image) # model A
        output = model(sat_image)["out"] # model B
        train_loss = criterion(output, mask_image)
        train_loss_record.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
        scheduler.step() # model B


    mean_train_loss = sum(train_loss_record)/len(train_loss_record)
    print("Epoch:", epoch)
    print('Train loss: {:.6f}'.format( mean_train_loss))

    model.eval()
    eval_loss = 0
    eval_loss_record = []
    eval_correct = 0

    with torch.no_grad():
        for i, (sat_image, mask_image) in enumerate(tqdm(valid_loader)):
            sat_image, mask_image = sat_image.to(device), mask_image.to(device)

            #output = model(sat_image) # model A
            output = model(sat_image)["out"] # model B
            eval_loss = criterion(output, mask_image)
            eval_loss_record.append(eval_loss.item())
    
    mean_train_loss = sum(train_loss_record)/len(train_loss_record)
    mean_eval_loss = sum(eval_loss_record)/len(eval_loss_record)
    print("Epoch:", epoch)
    print('Train loss: {:.6f}'.format( mean_train_loss))
    print('Evaluate loss: {:.6f}'.format(mean_eval_loss))
                
         
    if mean_eval_loss < best_loss:
        best_loss = mean_eval_loss
        print('Best loss:{:.6f} and save model.'.format(best_loss))
        torch.save(model.state_dict(), "p2_b.pth")
        
    print('The best loss is {:.6f} '.format(best_loss))
    
    '''
    # model B
    if epoch == 0:
        torch.save(model.state_dict(), "p2_b_0epo.pth")
    if epoch == 10:
        torch.save(model.state_dict(), "p2_b_10epo.pth")  
    if epoch == 20:
        torch.save(model.state_dict(), "p2_b_20epo.pth")
    '''