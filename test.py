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
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
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
classes = 0

def family_index():
    global classes
    with open(attribute_list) as f:
        dat = f.readlines()
    for i in range(len(dat)):
        dat[i] = dat[i].replace('\n','')
    classes = len(dat)
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
    ])), batch_size=64, shuffle=True, num_workers=4)

val_loader = torch.utils.data.DataLoader(
    ImageFilelist(root="./aircraft/data/images/", flist="./aircraft/data/images_"+attribute_name+"_val.txt",
                  transform=transforms.Compose([transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])), batch_size=64, shuffle=False, num_workers=1)

test_loader = torch.utils.data.DataLoader(
    ImageFilelist(root="./aircraft/data/images/", flist="./aircraft/data/images_"+attribute_name+"_test.txt",
                  transform=transforms.Compose([transforms.Scale((256,256)),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])), batch_size=16, shuffle=False, num_workers=1)

# TESTING

def test_model(network, criterion, testLoader, use_gpu = False, top_n = 1):
# Make a pass over the validation data.
    correct = 0.0
    cum_loss = 0.0
    counter = 0.0
    class_frq = [0]*70
    y_preds = []
    y_actual = []
    class_acc = [0]*70
    t = tqdm(testLoader, desc = 'Testing Model')
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
        max_scores, max_labels = outputs.data.max(1)
	correct += (max_labels == labels.data).sum()
	#max_k_scores, max_k_labels = torch.topk(outputs.data,top_n)
        #correct += (labels.data.view(-1,1) == max_k_labels).sum()
	for i in range(len(max_labels)):
	    y_preds.append(max_labels[i])
	    y_actual.append(labels.data[i])
	    class_frq[labels.data[i]] += 1
            if labels.data[i] == max_labels[i]:
                class_acc[labels.data[i]] += 1
        t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)   
    return (class_frq, class_acc, y_preds, y_actual)

# load the saved model
network = torch.load('resnet50_rescale_finetune_3.pth')
criterion = nn.CrossEntropyLoss()

# test top-1 accuracy
class_frq, class_acc, y_preds, y_actual = test_model(network, criterion, test_loader, use_gpu = True, top_n = 1)

results = [float(x)/float(y) for x, y in zip(class_acc, class_frq)]
classes = family_index()

#for i in range(len(classes)):
#    print(str(classes[i]) + ': %.3f\n' % (results[i]*100))

print('predictions')
for i in range(len(y_preds)):
    print(y_preds[i])
print('actual')
for i in range(len(y_actual)):
    print(y_actual[i])

"""
# test top-2 accuracy
class_acc = test_model(network, criterion, test_loader, use_gpu = True, top_n = 2)
print(class_acc)
# test top-3 accuracy
class_acc = test_model(network, criterion, test_loader, use_gpu = True, top_n = 3)
print(class_acc)
# test top-4 accuracy
class_acc = test_model(network, criterion, test_loader, use_gpu = True, top_n = 4)
print(class_acc)
# test top-5 accuracy
class_acc = test_model(network, criterion, test_loader, use_gpu = True, top_n = 5)
print(class_acc)
"""
