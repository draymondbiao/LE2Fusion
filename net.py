from model_LE2Fusion import Illumination_classifier
EPSILON = 1e-10
from torch import nn
import numpy as np
from data_loader.pixel_intensity_loss import con2

from data_loader.common import reflect_conv

import torch




class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))#np.floot:向下取整
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)# 对四周都填充 reflection_padding 行

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)#先填充
        out = self.conv2d(out)#卷积层
        if self.is_last is False:
            # out = F.normalize(out)
            # out = F.relu(out, inplace=False)#激活函数
            active = nn.LeakyReLU(inplace=False)
            out=active(out)
            # out = self.dropout(out)
        return out
class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        #组合级联
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    else:
        pass

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x
def toZeroThreshold(x, t=0.1):
    zeros = torch.cuda.FloatTensor(x.shape).fill_(0.0).cuda()
    return torch.where(x > t, x, zeros)

def Fusion_layer(vi_out, ir_out,vis_image):#中途融合方式（即串联）
    cls = Illumination_classifier(3)
    cls.cuda()
    cls.eval()
    out = cls(vis_image)
    out_origin = torch.tensor_split(out, 2, dim=1)#亮的地方和暗的地方
    out_ir = out_origin[0]
    out_vi = out_origin[1]
    abs_ir=torch.abs(out_ir)
    abs_vi=torch.abs(out_vi)

    ei_ir = con2(abs_ir)
    ei_vi = con2(abs_vi)

    ir_weight = ei_ir / (ei_ir + ei_vi+EPSILON)
    vi_weight = ei_vi / (ei_ir + ei_vi+EPSILON)

    tensor_f = torch.cat([vi_weight * vi_out+vi_out , ir_weight * ir_out+ir_out], dim=1)
    return tensor_f

def Fusion_weight(vis_image):
    cls = Illumination_classifier(3)
    cls.cuda()
    cls.eval()
    out = cls(vis_image)
    out_origin = torch.tensor_split(out, 2, dim=1)#亮的地方和暗的地方
    out_ir = out_origin[0]
    out_vi = out_origin[1]
    abs_ir=torch.abs(out_ir)
    abs_vi=torch.abs(out_vi)

    ei_ir = con2(abs_ir)
    ei_vi = con2(abs_vi)

    ir_weight = ei_ir / (ei_ir + ei_vi+EPSILON)
    vi_weight = ei_vi / (ei_ir + ei_vi+EPSILON)

    tensor_f = torch.cat([vi_weight , ir_weight], dim=1)
    return tensor_f

def Fusion_Ablation(vi_out, ir_out,vis_image):
    return torch.cat([vi_out, ir_out], dim=1)

# 渐进式融合网络架构  由特征提取器和图像重建器组成
class Encoder(nn.Module):#特征提取器
    # 从红外和可见光图像中完全 提取共同和互补 的特征
    def __init__(self):
        super(Encoder, self).__init__()
        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)

        self.conv1 = nn.Conv2d(1, 16, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(1, 16, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(1, 16, (5, 5), (1, 1), (2, 2))
        # self.att = AttentionNet(in_channels=1, out_channels=16, ngf=64, norm='IN', activation='lrelu')#in_channels, out_channels, ngf, norm, activation
        denseblock = DenseBlock
        self.DB1 = denseblock(16, 3, 1)


    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        # ac=nn.GELU()
        # sigmoid = nn.Sigmoid()
        # gap = nn.AdaptiveAvgPool2d(1)

        temx11 = activate(self.conv3(y_vi_image))
        temx1 = activate(self.conv1(y_vi_image) * (1 + temx11))

        temy21 = activate(self.conv3(ir_image))
        temy2 = activate(self.conv1(ir_image) * (1 + temy21))

        vi_out, ir_out = activate(self.DB1(temx1)), activate(self.DB1(temy2))

        return vi_out, ir_out


class Decoder(nn.Module):#图像重建器
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = reflect_conv(in_channels=128, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv4 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=1, stride=1, padding=0)

        self.convw1=nn.Conv2d(in_channels=2, kernel_size=1, out_channels=128, stride=1, padding=0)
        self.convw2 = nn.Conv2d(in_channels=128, kernel_size=1, out_channels=64, stride=1, padding=0)
        self.convw3 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=32, stride=1, padding=0)
        self.convw4 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x, weight):
        sigmoid = nn.Sigmoid()


        activate = nn.LeakyReLU()
        x = activate(self.conv1(x))
        weight = self.convw1(weight)
        weight = sigmoid(weight)
        x = x*weight+x

        x = activate(self.conv2(x))
        weight = self.convw2(weight)
        weight = sigmoid(weight)
        x = x * weight + x
        x = activate(self.conv3(x))
        weight = self.convw3(weight)
        weight = sigmoid(weight)
        x = x * weight + x

        x = nn.Tanh()(self.conv4(x)) / 2 + 0.5
        weight = self.convw4(weight)
        weight = sigmoid(weight)
        x = x*weight+x# 将范围从[-1,1]转换为[0,1]
        return x


class LE2Fusion(nn.Module):
    def __init__(self):
        super(LE2Fusion, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, y_vi_image, ir_image,vis_image):
        s=vis_image[:,:1]
        vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)#特征提取器
        # encoder_out = Fusion(vi_encoder_out, ir_encoder_out, vis_image)#torch.cat

        encoder_out = Fusion_layer(vi_encoder_out, ir_encoder_out, vis_image)
        decoder_weight=Fusion_weight(vis_image)
        # 将从红外和可见光图像中提取的深层特征连接起来，作为图像重建器的输入
        fused_image = self.decoder(encoder_out,decoder_weight)#图像重建器
        return fused_image

