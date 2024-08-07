import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset, DataLoader
import os
import torch.nn as nn
import torchvision.models as models
import timm
from dataset import ImageDataset
from model import Net
from types import SimpleNamespace
from timm.optim.optim_factory import create_optimizer
import pickle

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" 
#print(device)
torch.cuda.empty_cache()

# set random seed
seed = 325
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

train_transform = transforms.Compose([
    transforms.ToTensor(), 
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

'''
# model A epoch=20, ACC=50%
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
args = SimpleNamespace()
args.lr = 1e-3
args.weight_decay = 0
args.opt = 'adam'
args.momentum = 0.9
optimizer = create_optimizer(args, model)
epochs = 20
batch_size = 32
'''

# model B epoch=35, ACC=96%
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=True) # model B
model.fc = nn.Linear(in_features=64, out_features=50, bias=True)
#print(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
args = SimpleNamespace()
args.lr = 1e-4
args.weight_decay = 0
args.opt = 'sgd'
args.momentum = 0.9
optimizer = create_optimizer(args, model)
epochs = 100
batch_size = 32

dataset_path_train = "./hw1_data/p1_data/train_50"
dataset_path_val = "./hw1_data/p1_data/val_50"

train_set = ImageDataset(dataset_path_train, transform=train_transform)
valid_set = ImageDataset(dataset_path_val, transform=test_transform)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

best_acc = 0
model_scripted = torch.jit.script(model)

for epoch in range(epochs):

    model.train()
    train_loss = 0
    train_loss_record = []
    train_correct = 0

    for i, (image, label) in enumerate(tqdm(train_loader)):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        #_, output = model(image) # model A
        output = model(image) # model B
        train_loss = criterion(output, label)
        train_loss_record.append(train_loss.item())             
        pred_label = torch.max(output.data, 1)[1]
        train_correct = train_correct + pred_label.eq(label.view_as(pred_label)).sum().item()
        train_loss.backward()
        optimizer.step()

    model.eval()
    eval_loss = 0
    eval_loss_record = []
    eval_correct = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(valid_loader)):
            image, label = image.to(device), label.to(device)
            #_, output = model(image) # momdel A
            output = model(image) # model B
            eval_loss = criterion(output, label)
            eval_loss_record.append(eval_loss.item())
            pred_label = torch.max(output.data, 1)[1]
            eval_correct = eval_correct + pred_label.eq(label.view_as(pred_label)).sum().item()
    
    train_acc = 100 * train_correct / len(train_set)
    mean_train_loss = sum(train_loss_record)/len(train_loss_record)
    valid_acc = 100 * eval_correct / len(valid_set)
    mean_eval_loss = sum(eval_loss_record)/len(eval_loss_record)
    print("Epoch:", epoch)
    print('Train loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format( mean_train_loss, train_correct, len(train_set), train_acc))
    print('Evaluate loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(mean_eval_loss, eval_correct, len(valid_set), valid_acc))     
         
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('This epoch has best accuracy is {:.0f}% and save model'.format(best_acc))
        model_scripted.save('p1_b.pth')
        #torch.save(model.state_dict(), "p1_b.pth")


    print('The best accuracy is {:.0f}% '.format(best_acc))
    '''
    # model A
    if epoch == 0:
        torch.save(model.state_dict(), "p1_a_0epo.pth")
    if epoch == 4:
        torch.save(model.state_dict(), "p1_a_4epo.pth")  
    if epoch == 8:
        torch.save(model.state_dict(), "p1_a_8epo.pth")
    '''