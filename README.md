# TRAINING

```run.py```:  a simple Python script that can be run on any machine. So far, I've 
been running it on the department SLURM cluster, which provides all CS students access to 5 Tesla k20c GPUs, but
it can also be run on an AWS GPU instance with the proper packages installed (i.e., torch).  It imports necessary packages,
loads the training, testing, and validation datasets, and performs 10 training epochs using one of the various models we've
learned in class.  (Browse the end of the file to see where you need to edit learning rate/optimizer/etc for each model.)


```run.py``` utilizes a custom DataLoader to read image names and targets from text files, which this project requires.
I edited the aircraft .txt files such that all multi-word targets (e.g., "Boeing 747") are now single words ("Boeing_747")
Right now, ```run.py``` is set to train _only_ on family---to train on manufacturer, you'll need to change the following sections:

Line 25:  change ```attribute_list = "./aircraft/data/families.txt"``` to ```attribute_list = "./aircraft/data/manufacturers.txt"```

Line 28:  change ```attribute_name = "family"``` to ```attribute_name = "manufacturer"```

In order to train, you'll need to comment-out all models except the one you're testing (listed at end of ```run.py```).  
I know that ResNet and AlexNet both train successfully (75% and 25% validation accuracy from a cursory exploration).  VGG, DenseNet, and
GoogLeNet might need extra work and/or an array of GPUs to function properly, since they are very deep and quickly surpass 5GB memory.
