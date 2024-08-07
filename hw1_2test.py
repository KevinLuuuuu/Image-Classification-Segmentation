from tokenize import String
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
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path

def main(args):

    '''
    # model A  
    # model = VGG16_FCN32s().to(device) 
    '''

    # model B
    model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
    model.aux_classifier = None
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, 7).to(device)

    dataset_path = args.input_dir #"./hw1_data/p2_data/validation"
    output_path = args.output_dir #"./output"

    ckpt_path = "./p2_b.pth"
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.to(device)

    model.eval()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    batch_size = 1

    test_set = SegmentationDataset(dataset_path, transform=test_transform, train_set=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def process_output(output, mask_name):
        allbatch_max_index = output["out"].max(1)[1].cpu() # model B
        mask_image = np.zeros((512, 512, 3), dtype=np.uint8)
        for max_index, name in zip(allbatch_max_index, mask_name):
            mask_image = np.zeros((512, 512, 3), dtype=np.uint8)

            mask_image[max_index == 0] = [0, 1, 1]
            mask_image[max_index == 1] = [1, 1, 0]
            mask_image[max_index == 2] = [1, 0, 1]
            mask_image[max_index == 3] = [0, 1, 0]
            mask_image[max_index == 4] = [0, 0, 1]
            mask_image[max_index == 5] = [1, 1, 1]

            mask_image = mask_image*255
            img = Image.fromarray(mask_image, 'RGB')
            img.save(str(output_path) + name)

    with torch.no_grad():
        for image, mask_name in tqdm(test_loader):
            image = image.to(device)
            output = model(image)
            process_output(output, mask_name)


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