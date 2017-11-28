import os
from pytorch_sift import SIFTNet
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
import cv2
import time
import numpy as np


patch_size = 65
ON_GPU = False
SIFT = SIFTNet()
SIFT.eval()
if ON_GPU:
    SIFT.cuda()
    
image = cv2.imread('img/fox.png',0)
plt.figure()
plt.imshow(image, cmap="gray")

h,w = image.shape
patch = image[h/4: h/4 + patch_size,w/3 : w/3 + patch_size]
plt.figure()
plt.imshow(patch, cmap="gray")

# It takes n_patches x 1 x patch_size x patch_size input
patches = np.ndarray((1, 1, patch_size, patch_size), dtype=np.float32)
patches[0,0,:,:] = patch


t = time.time()
if ON_GPU:
     Variable(torch.from_numpy(patches).cuda(), volatile=True)
else:
    torch_patches = Variable(torch.from_numpy(patches), volatile=True)

res = SIFT(torch_patches)
sift = np.round(512. * res.data.cpu().numpy()).astype(np.int32)
print 'time', time.time() - t
print sift