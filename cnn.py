import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

data_dir = "input/seg_train/seg_train/"
test_data_dir = "input/seg_test/seg_test"

dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

img, label = dataset[0]
print(img.shape,label)