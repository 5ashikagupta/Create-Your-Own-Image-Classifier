'''python train.py data_directory

#prints out training loss

#prints out validation loss

#prints out validation accuracy

#set directory to save checkpoints
python train.py data_dir --save_dir save_directory
python train.py data_dir --arch "vgg13"
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
python train.py data_dir --gpu'''


import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict





data_dir = ' '
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Defining transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.299, 0.224, 0.225))])

#Loading the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


#Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)


parser = argparse.ArgumentParser(
    description = 'Parser for train.py'
)
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--gpu', action="store", default="gpu")

args = parser.parse_args()
where = args.data_dir
path = args.save_dir
lr = args.learning_rate
struct = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
dropout = args.dropout

#train model
    import torchvision.models as models
    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(25088, 4096)),
        ("relu1", nn.ReLU()),
        ("dropout1", nn.Dropout(p=0.5)),
        ("fc2", nn.Linear(4096, 1024)),
        ("relu2", nn.ReLU()),
        ("dropout2", nn.Dropout(p=0.5)),
        ("fc3", nn.Linear(1024, 102)),
        ("logsoftmax", nn.LogSoftmax(dim=1))
    ]))
            model.classifier = classifier   

    elif args.arch == "Densenet":
        model = models.densenet121(pretrained=True)
                                  #parameters to be trained, features are frozen
        for param in model.parameters():
            param.requires_grad = False
            classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(25088, 4096)),
        ("relu1", nn.ReLU()),
        ("dropout1", nn.Dropout(p=0.5)),
        ("fc2", nn.Linear(4096, 1024)),
        ("relu2", nn.ReLU()),
        ("dropout2", nn.Dropout(p=0.5)),
        ("fc3", nn.Linear(1024, 102)),
        ("logsoftmax", nn.LogSoftmax(dim=1))
    ]))

        model.classifier = classifier                              
        model.classifier = checkpoint["classifier"]
        model.load_state_dict(checkpoint["state_dict"])
        model.class_to_idx = checkpoint["class_to_idx"]

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    epochs = args.epochs


    device =  torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
def main():
    for e in range(epochs):
        running_loss = 0
        train_loss = 0



        for images, labels in train_loader:

            log_ps = model(images)
            train_loss += criterion(log_ps, labels)
            optimizer.zero_grad
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()

            model.train()

            with torch.no_grad():
                valid_loss = 0
                accuracy = 0
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        log_ps = model(images)
        valid_loss += criterion(log_ps, labels).item()

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    avg_valid_loss = valid_loss / len(valid_loader)
    avg_accuracy = accuracy / len(valid_loader)


    print("Average Train Loss:", avg_train_loss)
    print("Average Valid Loss:", avg_valid_loss)
    print("Average Accuracy:", avg_accuracy)

    torch.save({
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }, args.save_dir + '/checkpoint.pth')

if __name__ == "__main__":
    main()
