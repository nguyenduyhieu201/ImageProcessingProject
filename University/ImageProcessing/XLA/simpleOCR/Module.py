import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os

from PIL import Image
data_dir = 'data'
TRAIN = 'train'
VAL = 'val'

data_transforms = {
    TRAIN: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    )
    for x in [TRAIN, VAL]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL]}
for x in [TRAIN, VAL]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

# print("Classes: ")
class_names = image_datasets[TRAIN].classes
# print(image_datasets[TRAIN].classes)

idx_to_class = {v: k for k, v in image_datasets[TRAIN].class_to_idx.items()}
print(idx_to_class)

# Tinh chỉnh chuyển đổi
# Load the pretrained model from pytorch
resnet50 = models.resnet50(pretrained=True)

# Freeze training for all layers
for param in resnet50.parameters():
    param.require_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_features, len(class_names))

resnet50.load_state_dict(torch.load("data/ResNet_50_50.pt", map_location=torch.device('cpu')))

def predict(img_path, model =resnet50):
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path)

    test_image_tensor = transform(img)
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)
        return idx_to_class[topclass.cpu().numpy()[0][0]]
