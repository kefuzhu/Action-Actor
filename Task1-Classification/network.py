import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math
# Extra library
import torch.nn.functional as F

def GlobalAvgPool(x):
    x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
    return x

# 4-convolution layers
class net(nn.Module):
    ####
    # define your model
    ####
	def __init__(self):
		super(net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 2)
		self.conv2 = nn.Conv2d(6, 16, 2)
		self.conv3 = nn.Conv2d(16, 32, 2)
		self.conv4 = nn.Conv2d(32, 64, 2)
		self.fc1 = nn.Linear(64, 43) # Number of classes:43

	def forward(self, x):
		in_size = x.size(0)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = GlobalAvgPool(x)
		x = x.view(in_size, -1)
		x = self.fc1(x)
		return F.log_softmax(x, dim = 1)
