# AEROVISION

### Organization
+ The `aircraft` directory contains all images in addition to text files denoting the training, validation, and test sets.

+ The `unused` directory contains training code and saved models from the model selection stage of the project.

+ The `hand_test` directory contains manually-obtained test images from Google image search.  These were used in conjunction with the test set to assess performance on new data.

+ The `output` directory contains various `.err` files with training/validation output from the cluster (necessary because notebook port forwarding doesn't work across the job scheduler)

+ `vision_github.ipynb` contains code for dataloader construction, matplotlib rendering, and test classifications.  It also includes our training/validation/testing loops, though these are run externally from various Python files.

+ `classify_utils.py` contains code obtained from the Internet that formats the sklearn classification report using PyPlot (reasoning: the raw output from sklearn is a text array) Ref:  https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

+ `resnet50_rescale_finetune_3.pth` is our most up-to-date trained model (used in application backend).

+ Various bash scripts were used to run training and testing code on the CS department slurm cluster via the command `sbatch <filename>.sh`.

+ Various text files list the output _prediction_ and _ground-truth_ results generated during testing.  These were fed into an sklearn function to produce the classification report on page 5 of the final report.
