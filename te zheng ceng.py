import os

import cv2
import torch

from Ultra.net.model import SAFMN
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

#图像预处理，要与生成alexnet.pth文件的train预处理一致
data_transform = transforms.Compose(
    [transforms.Resize((512, 512)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# data_transform = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
# model = DEDCGCNEE(n_classes=2)
model = SAFMN(dim=48, n_blocks=8, ffn_scale=2.0)
# load model weights
# model_weight_path = "D://Money-make/home/tmpusr/Desktop/wuxuelong/Code/UVM-Net-main/Ultra/Resultd//unet-pytorch-main//logs//best_epoch_weights.pth"  # "./resNet34.pth"
# model.load_state_dict(torch.load(model_weight_path))
print(model)

# load image
img = Image.open('/home/tmpusr/Desktop/wuxuelong/Code/UVM-Net-main/Ultra/data/Test/LOL/high/2.png')
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)

for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]print(model)
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)#行，列，索引
        # [H, W, C]
        plt.imshow(im[:, :, i])#cmap默认为蓝绿图
    plt.savefig('a3.png', dpi=1000)

def feature_vis(e3):  # feaats形状: [b,c,h,w]
    output_shape = (512, 512)  # 输出形状
    channel_mean = torch.mean(e3, dim=1, keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
    channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
    channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
    channel_mean = (
                ((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(
        np.uint8)
    savedir = '/home/tmpusr/Desktop/wuxuelong/Code/UVM-Net-main/Ultra/Result'
    if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
    channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    cv2.imwrite(savedir + 'feature_vis/' + '0.png', channel_mean)


