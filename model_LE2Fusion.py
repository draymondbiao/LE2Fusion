import math
from torch import nn



class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

# nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
class LE2(nn.Module):
    def __init__(self, input_channels, init_weights=True):
        super(LE2, self).__init__()

        self.conv1 = Conv1(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.conv2 = Conv1(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv3 = Conv1(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv4 = Conv1(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv5 = Conv1(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv6 = Conv1(in_channels=32, out_channels=2, kernel_size=3, padding=1, stride=1)


        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化权重

        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        activate = nn.LeakyReLU(inplace=False)
        # x = self.conv1(x)

        x = activate(self.conv1(x))
        # x_DB = self.DB1(x)
        x = activate(self.conv2(x))

        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = activate(self.conv5(x))
        x = activate(self.conv6(x))
        # x = activate(self.conv7(x))
        # x = activate(self.conv8(x))
        return x




