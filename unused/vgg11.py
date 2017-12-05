##########################################################################
# IMPORT PACKAGES ########################################################
##########################################################################

import torch, random
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm as tqdm
import os as os
import os.path
from PIL import Image

##########################################################################
# EDIT HERE TO CHANGE TARGET ATTRIBUTE ###################################
##########################################################################

# families, manufacturers
attribute_list = "./aircraft/data/families.txt"

# family, manufacturer
attribute_name = "family"

##########################################################################
# DATA LOADING ###########################################################
##########################################################################

# number of unique classes
classes = 70

def family_index():
    global classes
    with open(attribute_list) as f:
        dat = f.readlines()
    for i in range(len(dat)):
        dat[i] = dat[i].replace('\n','')
    return dat

# from PyTorch GitHub:  https://github.com/pytorch/vision/issues/81  

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    imlist = []
    fams = family_index()
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append( ((''.join([impath,'.jpg'])), fams.index(imlabel)) )
    return imlist

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, 
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root   = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root,impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imlist)

##########################################################################
# TRAINING, VALIDATION, and TESTING ######################################
##########################################################################

train_loader = torch.utils.data.DataLoader(
    ImageFilelist(root="./aircraft/data/images/", flist="./aircraft/data/images_"+attribute_name+"_train.txt",
                  transform=transforms.Compose([transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])), batch_size=4, shuffle=True, num_workers=4)

val_loader = torch.utils.data.DataLoader(
    ImageFilelist(root="./aircraft/data/images/", flist="./aircraft/data/images_"+attribute_name+"_val.txt",
                  transform=transforms.Compose([transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])), batch_size=4, shuffle=False, num_workers=1)

test_loader = torch.utils.data.DataLoader(
    ImageFilelist(root="./aircraft/data/images/", flist="./aircraft/data/images_"+attribute_name+"_test.txt",
                  transform=transforms.Compose([transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])), batch_size=32, shuffle=False, num_workers=1)


##########################################################################
# TRAIN ##################################################################
##########################################################################

def train_model(network, criterion, optimizer, scheduler, trainLoader, valLoader, n_epochs = 10, use_gpu = False):
    if use_gpu:
        network = network.cuda()
        criterion = criterion.cuda()
        
    # Training loop.
    for epoch in range(0, n_epochs):
        correct = 0.0
        cum_loss = 0.0
        counter = 0.0
        
        scheduler.step()
        
        # Make a pass over the training data.
        t = tqdm(trainLoader, desc = 'Training epoch %d' % epoch)
        network.train()  # This is important to call before training!
        for (i, (inputs, labels)) in enumerate(t):
            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # Backward pass:
            optimizer.zero_grad()
            # Loss is a variable, and calling backward on a Variable will
            # compute all the gradients that lead to that Variable taking on its
            # current value.
            loss.backward() 

            # Weight and bias updates.
            optimizer.step()

            # logging information.
            counter += inputs.size(0)
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)

        # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        counter = 0.0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
        network.eval()  # This is important to call before evaluating!
        for (i, (inputs, labels)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # logging information.
            counter += inputs.size(0)
            cum_loss += loss.data[0]
            #max_k_scores, max_k_labels = torch.topk(outputs.data,3)
            #correct += (labels.data.view(-1,1) == max_k_labels).sum()
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)   



##########################################################################
# MODELS #################################################################
##########################################################################

# VGG11
network = models.vgg11(pretrained=True)
mod = list(network.classifier.children())
mod.pop()
mod.append(nn.Linear(4096, classes))
new_classifier = nn.Sequential(*mod)
network.classifier = new_classifier
learningRate = 5e-4 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(network.parameters(), lr = learningRate, weight_decay = 0.001)
scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[4,9,15],gamma=0.2);

##########################################################################
# START TRAINING #########################################################
##########################################################################

train_model(network, criterion, optimizer, scheduler, train_loader, val_loader, n_epochs = 20, use_gpu = True)
torch.save(network, 'vgg11_20e_5e-4.pth')
