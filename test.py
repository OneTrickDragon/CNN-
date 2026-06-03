import os
import json
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import configargparse
from cnn import ConvNet
from torchvision.transforms import transforms

def predict(dataset_pred, img_name, transformer):
    image = Image.open(dataset_pred + '/' + img_name)
    image_tensor = torch.unsqueeze(transformer(image), dim = 0)
    image_tensor = image_tensor.to(device)

    predictions = model(image_tensor)

    softmax = nn.Softmax(dim=1)
    predictions_softmax = softmax(predictions)
    predictions_softmax = torch.squeeze(predictions_softmax)
    predictions_softmax = predictions_softmax.cpu().detach().numpy()

    max_index = np.argmax(predictions_softmax)
    probability = predictions_softmax[max_index]
    predicted_class = classes[max_index]

    return predicted_class, probability, predictions_softmax


if __name__ == "__main__":