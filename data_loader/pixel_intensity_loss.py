import torch
from PIL import Image
from einops import rearrange
from torchvision import transforms
from torchvision.utils import save_image
from torch import optim, nn
import torch.nn.functional as F




def pixel_intensity(img):# pixel intensity
    b = img.size()[0]
    window=torch.ones(size=(1,1,3,3)).to(img.device)
    # window = torch.ones(size=(1, 1, 3, 3))
    from torch.autograd import Variable
    window = Variable(window.expand(1, 1, 3, 3).contiguous())
    con1=F.conv2d(img,window,padding=1)
    return con1

def ei(img):#(16,1,64,64)
    x1=img.size()[2]
    x11=int(x1/4)
    x2=img.size()[3]
    x22=int(x2/4)
    img = rearrange(img, 'b c (x1 y1) (x2 y2) ->b c (x1 x2) (y1 y2)',x1=x11,y1=4,x2=x22,y2=4)  #
    l1 = torch.norm(img,p=1,dim=3,keepdim=True)
    l1 = torch.repeat_interleave(l1,repeats=16,dim=-1)
    l1 = rearrange(l1,'b c (x1 x2) (y1 y2) -> b c (x1 y1) (x2 y2) ',x1=x11,y1=4,x2=x22,y2=4)
    return l1
# save_image(img,"1.png")
# print(img1)
# ei1=con2(img1)
# print(ei1)

# print(s[1][0].shape)

def max3x3(a,b,r,c):
    max3=a
    for i in range(0,r-3,3):
        for j in range(0,c-3,3):
            if (a[i][j]+a[i][j+1]+a[i][j+2]+a[i+1][j]+a[i+1][j+1]+a[i+1][j+2]+a[i+2][j]+a[i+2][j+1]+a[i+3][j+2]
                    >= b[i][j]+b[i][j+1]+b[i][j+2]+b[i+1][j]+b[i+1][j+1]+b[i+1][j+2]+b[i+2][j]+b[i+2][j+1]+b[i+3][j+2]):
                max3[i][j]=a[i][j]
                max3[i][j + 1]=a[i][j + 1]
                max3[i][j + 2] = a[i][j + 2]
                max3[i + 1][j] = a[i + 1][j]
                max3[i + 1][j + 1] = a[i + 1][j + 1]
                max3[i + 1][j + 2] =a[i + 1][j + 2]
                max3[i + 2][j] =a[i + 2][j]
                max3[i + 2][j + 1] = a[i + 2][j + 1]
                max3[i + 3][j + 2] =a[i + 3][j + 2]
            else:
                max3[i][j] = b[i][j]
                max3[i][j + 1] = b[i][j + 1]
                max3[i][j + 2] = b[i][j + 2]
                max3[i + 1][j] = b[i + 1][j]
                max3[i + 1][j + 1] = b[i + 1][j + 1]
                max3[i + 1][j + 2] = b[i + 1][j + 2]
                max3[i + 2][j] = b[i + 2][j]
                max3[i + 2][j + 1] = b[i + 2][j + 1]
                max3[i + 3][j + 2] = b[i + 3][j + 2]
    return max3
# print(img1[0][0].shape,img2[0][0].shape)
# max3=max3x3(img1[0][0],img2[0][0],585,426)
# print(max3)

