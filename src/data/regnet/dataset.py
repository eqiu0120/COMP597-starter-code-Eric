import os
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms

def load_data(conf) -> data.Dataset:
    root = f"/home/slurm/comp597/students/{os.environ['USER']}/fake_imagenet/FakeImageNetSmall/train"
    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    return ImageFolder(root=root, transform=tfm)