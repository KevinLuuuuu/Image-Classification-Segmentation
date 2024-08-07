import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import timm
from dataset import ImageDataset
from model import Net
import numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" 
#print(device)
torch.cuda.empty_cache()

ckpt_path = "./p1_a_0epo.pth"
dataset_path = "./hw1_data/p1_data/val_50"

# model A
model = Net()

'''
# model B
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg13_bn", pretrained=True) # model B
model.classifier[6] = nn.Linear(in_features=512, out_features=50, bias=True)
'''

ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt)
model = model.to(device)

model.eval()

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 10
test_set = ImageDataset(dataset_path, transform=test_transform, train_set=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

representation, label_list = [], []

with torch.no_grad():
    for i, (image, label) in enumerate(tqdm(test_loader)):
        image = image.to(device)
        repre,_ = model(image) # model A
        #repre = model(image) # model B
        for rep, lab in zip(repre, label):
            rep = rep.reshape(-1)
            representation.append(rep.cpu().numpy())
            label_list.append(lab)

rep_arr = numpy.array(representation)
lab_arr = numpy.array(label_list)

# prepare color
color = plt.get_cmap('gist_ncar')
colors = []
for i in range(50):
    colors.append(color(i/50))
plt.figure(figsize=(10, 10))

# PCA
XY_pca = PCA(n_components=2, random_state=5203).fit_transform(rep_arr)
XY_pca_min = XY_pca.min(0)
XY_pca_max = XY_pca.max(0)
XY_norm = (XY_pca - XY_pca_min)/(XY_pca_max - XY_pca_min)

for i in range(XY_norm.shape[0]):
    plt.plot(XY_norm[i, 0], XY_norm[i, 1], 'o', color=colors[lab_arr[i]])
plt.savefig('PCA0.png')

# t-sne
XY_tsne = TSNE(n_components=2, init='random', random_state=5203).fit_transform(rep_arr)
XY_tsne_min = XY_tsne.min(0) 
XY_tsne_max = XY_tsne.max(0)
XY_norm = (XY_tsne - XY_tsne_min)/(XY_tsne_max - XY_tsne_min)

for i in range(XY_norm.shape[0]):
    plt.plot(XY_norm[i, 0], XY_norm[i, 1], 'o', color=colors[lab_arr[i]])
plt.savefig('TSNE0.png')