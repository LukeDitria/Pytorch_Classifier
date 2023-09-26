import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torchvision.models as model
import torchvision

import os
from tqdm import trange
import argparse
import matplotlib.pyplot as plt

from Trainer import ModelTrainer
import Helpers as hf

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=True)

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
parser.add_argument("--model_type", "-mt", help="Pytorch Classifier model to use!", type=str,
                    default="resnet18")

parser.add_argument("--model_parameters", "-mp", help="Pytorch Classifier Parameters to use!", type=str,
                    default=None)

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=100)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=32)
parser.add_argument("--image_size", help="Input image size", type=int, default=128)
parser.add_argument("--output_size", help="Number of output classes", type=int, default=None)

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
# float args
parser.add_argument("--learning_rate", "-lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--split",
                    help="Percentage of Training images used for validation if a validation set is not provided",
                    type=float, default=0.1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--test_trainset", action='store_true',
                    help="Get the end-of-epoch training accuracy as well as the continuous training accuracy")
parser.add_argument("--show_aug_images", action='store_true',
                    help="Display a batch of the augmented training images!")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")
print("")

print("-Image Size %d" % args.image_size)

train_transform = transforms.Compose([transforms.Resize(args.image_size),
                                      transforms.CenterCrop(args.image_size),
                                      transforms.AutoAugment(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(args.image_size),
                                     transforms.CenterCrop(args.image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

train_set, test_set, val_set, train_classes = hf.get_datasets(dataset_root=args.dataset_root,
                                                              split=args.split,
                                                              train_transform=train_transform,
                                                              test_transform=test_transform)

if args.output_size is not None:
    output_size = args.output_size
else:
    output_size = train_classes

print("-Number of output classes %d" % output_size)
if args.model_parameters is None:
    print("-NOT USING PRE-TRAINED WEIGHTS!")
else:
    print("-Using pre-trained weights type %s" % args.model_parameters)

# Create classifier network.
classifier = model.get_model(args.model_type, weights=args.model_parameters).to(device)

# Replace the last Linear Layer with one that has the correct number of outputs!
# This should cover most of the current model types...
if hasattr(classifier, "fc"):
    num_ftrs = classifier.fc.in_features
    classifier.fc = nn.Linear(num_ftrs, output_size).to(device)
elif hasattr(classifier, "classifier"):
    if isinstance(classifier.classifier, nn.Linear):
        num_ftrs = classifier.classifier.in_features
        classifier.classifier = nn.Linear(num_ftrs, output_size).to(device)
    if isinstance(classifier.classifier, nn.Sequential):
        num_ftrs = classifier.classifier[-1].in_features
        classifier.classifier[-1] = nn.Linear(num_ftrs, output_size).to(device)
elif hasattr(classifier, "heads"):
    if isinstance(classifier.heads, nn.Linear):
        num_ftrs = classifier.heads.in_features
        classifier.heads = nn.Linear(num_ftrs, output_size).to(device)
    if isinstance(classifier.heads, nn.Sequential):
        num_ftrs = classifier.heads[-1].in_features
        classifier.heads[-1] = nn.Linear(num_ftrs, output_size).to(device)


# Let's see how many Parameters our Model has!
num_model_params = 0
for param in classifier.parameters():
    num_model_params += param.flatten().shape[0]
print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))

# Create Model Trainer object!
model_trainer = ModelTrainer(model=classifier, device=device, loss_fun=nn.CrossEntropyLoss(),
                             batch_size=args.batch_size, learning_rate=args.learning_rate,
                             save_dir=args.save_dir, model_name=args.model_name,
                             start_from_checkpoint=args.load_checkpoint)

# Set Data-set split of model!
model_trainer.set_data(train_set=train_set, test_set=test_set, val_set=val_set)

if args.show_aug_images:
    plt.figure(figsize=(20, 10))
    images, labels = next(iter(model_trainer.train_loader))
    out = torchvision.utils.make_grid(images[0:8], normalize=True)
    _ = plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.show()

# Implements our training loop
valid_acc = model_trainer.best_valid_acc
cont_train_acc = model_trainer.cont_train_acc_logger[-1] if len(model_trainer.cont_train_acc_logger) > 0 else 0
pbar = trange(model_trainer.start_epoch, args.nepoch, leave=False, desc="Epoch")
for epoch in pbar:
    pbar.set_postfix_str('Train: %.2f%%, Valid: %.2f%%' % (cont_train_acc * 100, valid_acc * 100))

    # Call the training function and pass training dataloader etc
    cont_train_acc = model_trainer.train_model()

    # Call the modules evaluate function for train and validation set
    if args.test_trainset:
        _ = model_trainer.evaluate_model(train_test_val="train")
    valid_acc = model_trainer.evaluate_model(train_test_val="val")

    # Check if the current validation accuracy is greater than the previous best
    # If so, then save the model
    if valid_acc > model_trainer.best_valid_acc:
        model_trainer.save_checkpoint(epoch, valid_acc)

test_acc = model_trainer.evaluate_model(train_test_val="test")
print("Final Test Accuracy: %.2f%%" % (test_acc * 100))
