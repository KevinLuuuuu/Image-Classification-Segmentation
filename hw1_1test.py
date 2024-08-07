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
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle

def main(args):

    dataset_path = args.input_dir #"./hw1_data/p1_data/val_50"
    output_path = args.output_dir #"./p1_output.csv"
    ckpt_path = "./p1_b.pth"
    
    #model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=True) # model B
    #model.fc = nn.Linear(in_features=64, out_features=50, bias=True)
    #print(type(model))
    #torch.save(model, "p1_b_arch.pth")
    #print(model)

    #model = torch.load('cifar100_vgg11_bn-57d0759e.pt')
    #print(type(model))
    #print(model)

    model = torch.jit.load('p1_b.pth')

    #ckpt = torch.load(ckpt_path)
    #model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    batch_size = 10
    test_set = ImageDataset(dataset_path, transform=test_transform, train_set=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    pred_label_list = []
    image_name_list = []


    with torch.no_grad():
        for i, (image, image_name) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            output = model(image)
            pred_label = torch.max(output.data, 1)[1]
            
            pred_label_list.append(pred_label)
            image_name_list.append(image_name)

    ################# check ####################
    with open(output_path, 'w', newline="") as fp:        
        file_writer = csv.writer(fp)
        file_writer.writerow(['filename', 'label'])
        for i in range(len(pred_label_list)):
            for j in range(len(pred_label_list[i])):
                file_writer.writerow([image_name_list[i][j], pred_label_list[i][j].item()])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Path to the input file.",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the input file.",
        required=True
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu" 
    #print(device)
    torch.cuda.empty_cache()

    args = parse_args()
    main(args)
                 