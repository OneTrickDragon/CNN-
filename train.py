import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import configargparse
from cnn import ConvNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils import plot_loss_acc

def compute_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def create_model_folder(log_dir):
    
    os.mkdir(log_dir)
    
    checkpoints_path = 'checkpoints/'
    checkpoints_path = os.path.join(log_dir, checkpoints_path)
    os.mkdir(checkpoints_path)

    return checkpoints_path

def train(model, train_loader, validation_loader, loss_fn, optimizer, num_epochs,
          batch_size, learning_rate, device, log_dir, checkpoints_path):
    
    max_val_acc = 0.0

    train_acc = []
    train_losses = []

    val_acc = []
    val_losses = []
    bestEpoch = 0

    print("-----------------------------------------------------")

    for i in range(num_epochs):

        train_acc = 0.0
        train_loss = 0.0

        print(f"Epoch {i+1}")
        model.train()
        iteration = 1

        print('\nTraining')

        for images, labels in train_loader:

            print('\rEpoch[' + str(i+1) + '/' + str(num_epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(train_loader)), end='')
            iteration += 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            loss.backward()
            optimizer.step()

            train_acc += compute_accuracy(predictions, labels).item()
            train_loss = loss.item()

        val_acc = 0.0
        val_loss = 0.0

        model.eval()
        iteration = 1

        print('')
        print('\nValidation')

        for images, labels in validation_loader:
            iteration += 1

            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            val_acc += compute_accuracy(predictions, labels).item()
            val_loss += loss.items()


        print(f'- Train Acc: {(train_acc / len(train_loader))*100:.2f}%')
        print(f'- Val Acc: {(val_acc / len(validation_loader))*100:.2f}%')
        print(f'- Train Loss: {train_loss / len(train_loader):.3f}')
        print(f'- Val Loss: {val_loss / len(validation_loader):.3f}')
        
        if i % 10 == 0:
            torch.save(model.state_dict(), checkpoints_path + "/checkpoint_" + str(i) + ".pth")

        if (val_acc / len(validation_loader)) > max_val_acc:

            if i == 0:
                torch.save(model.state_dict(), checkpoints_path + "checkpoint_" + str(i) + "_best.pth")
            else:
                os.remove(checkpoints_path + "checkpoint_" + str(bestEpoch) + "_best.pth")
                torch.save(model.state_dict(), checkpoints_path + "checkpoint_" + str(i) + "_best.pth")






