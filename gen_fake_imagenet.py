import os
from pathlib import Path
import torch
from torchvision.transforms import ToPILImage

root = Path('/home/slurm/comp597/students') / os.environ['USER'] / 'fake_imagenet' / 'FakeImageNetSmall'
train_dir = root / 'train'

num_classes = 1000
images_per_class = 2   # 2000 images total
H = W = 128

to_pil = ToPILImage()
train_dir.mkdir(parents=True, exist_ok=True)

for cls in range(num_classes):
    cls_dir = train_dir / str(cls)
    cls_dir.mkdir(parents=True, exist_ok=True)
    for j in range(images_per_class):
        img = torch.randint(0, 256, (3, H, W), dtype=torch.uint8)
        to_pil(img).save(cls_dir / f'{cls}_{j}.JPEG')

print("Created:", train_dir)
