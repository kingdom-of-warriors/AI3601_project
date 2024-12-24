import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Model for RM-ResNet

thresh = 0.5  # neuronal threshold
decay = 0.25  # decay constants
num_classes = 1000
time_window = 2


# membrane potential update

class mem_update(nn.Module):
    def __init__(self, act=False, inference=False, d=4):
        super(mem_update, self).__init__()
        self.act = act
        self.inference = inference
        self.qtrick = MultiSpike4() if d == 4 else MultiSpike8()
        
    def forward(self, x):
        if not self.inference:
            spike = torch.zeros_like(x[0]).to(x.device)
            output = torch.zeros_like(x)
            mem_old = 0
            time_window = x.shape[0]
            for i in range(time_window):
                if i >= 1:
                    mem = (mem_old - spike.detach()) * decay + x[i]

                else:
                    mem = x[i]
                spike = self.qtrick(mem)
                mem_old = mem.clone()
                output[i] = spike
            # print(output[0][0][0][0])
            return output
        else:
            pass


class MultiSpike8(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant8(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=8))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
#         print(self.quant8.apply(x))
        return self.quant8.apply(x)

class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

class batch_norm_2d(nn.Module):
    """TDBN"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class batch_norm_2d1(nn.Module):
    """TDBN-Zero init"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0)
            nn.init.zeros_(self.bias)


class Snn_Conv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 marker='b'):
        super(Snn_Conv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        weight = self.weight
        h = (input.size()[3] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        w = (input.size()[4] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        c1 = torch.zeros(time_window,
                         input.size()[1],
                         self.out_channels,
                         h,
                         w,
                         device=input.device)
        # print(111, c1.shape, weight.shape)
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        return c1


class BasicBlock_18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            mem_update(),
            Snn_Conv2d(in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=False),
            batch_norm_2d(out_channels),
            mem_update(),
            Snn_Conv2d(out_channels,
                       out_channels * BasicBlock_18.expansion,
                       kernel_size=3,
                       padding=1,
                       bias=False),
            batch_norm_2d1(out_channels * BasicBlock_18.expansion),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock_18.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels,
                           out_channels * BasicBlock_18.expansion,
                           kernel_size=1,
                           stride=stride,
                           bias=False),
                batch_norm_2d(out_channels * BasicBlock_18.expansion),
            )

    def forward(self, x):
        return (self.residual_function(x) + self.shortcut(x))


class ResNet_origin_18(nn.Module):
    # Channel:
    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3,
                       64 ,
                       kernel_size=7,
                       padding=3,
                       bias=False,
                       stride=2),
            batch_norm_2d(64),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.mem_update = mem_update()
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # print("in_channels", self.in_channels)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        input = torch.zeros(time_window,
                            x.size()[0],
                            3,
                            x.size()[2],
                            x.size()[3],
                            device=device)
        for i in range(time_window):
            input[i] = x
        output = self.conv2_x(input)    
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.mem_update(output)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.fc(output)
        return output


def resnet18(pretrained=False):
    model = ResNet_origin_18(BasicBlock_18, [2, 2, 2, 2])
    if pretrained:
        # 加载预训练权重
        new_state_dict = {}
        pretrained_weights = torch.load('/home/ljr/monodepth2/networks/resnet18.pth')
        for k, v in pretrained_weights.items():
            name = k.replace('module.', '')  # 移除 'module.' 前缀
            new_state_dict[name] = v  # 添加 'encoder.' 前缀
        model.load_state_dict(new_state_dict)
        print('成功加载预训练模型权重')
    return model

# from torchsummary import summary
# device = "cuda"
# model = resnet18().to(device)
# a = torch.randn(1, 3, 192, 640).to(device)
# b = model(a)
# print(b.shape)

class SResNetMultiImageInput(ResNet_origin_18):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, num_block, num_classes=1000, num_input_images=1):
        super(SResNetMultiImageInput, self).__init__(block, num_block)
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3 * num_input_images,
                       64,
                       kernel_size=7,
                       padding=3,
                       bias=False,
                       stride=2),
            batch_norm_2d(64),
        )
        # self.mem_update = mem_update()
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv2_x = self._make_layer(block, 64, num_block[0], stride=2)
        # self.conv3_x = self._make_layer(block, 128, num_block[1], stride=2)
        # self.conv4_x = self._make_layer(block, 256, num_block[2], stride=2)
        # self.conv5_x = self._make_layer(block, 512, num_block[3], stride=2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


def Sresnet_multiimage_input(pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    blocks = [2, 2, 2, 2]
    block_type = BasicBlock_18
    model = SResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    if pretrained:
        loaded = {}
        pretrained_weights = torch.load('/home/ljr/monodepth2/networks/resnet18.pth', map_location='auto')
        for k, v in pretrained_weights.items():
            name = k.replace('module.', '')  # 移除 'module.' 前缀
            loaded[name] = v  # 添加 'encoder.' 前缀
        loaded['conv1.0.weight'] = torch.cat(
            [loaded['conv1.0.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class SResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder"""
    def __init__(self, pretrained=False, num_input_images=1):
        super(SResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])


        if num_input_images > 1:
            self.encoder = Sresnet_multiimage_input(pretrained, num_input_images)
        else:
            self.encoder = resnet18(pretrained)

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = x.unsqueeze(0).repeat(time_window, 1, 1, 1, 1)
        x = self.encoder.conv1(x)
        x = self.encoder.mem_update(x)
        self.features.append(x.mean(0))
        x = self.encoder.conv2_x(x)
        self.features.append(x.mean(0))
        x = self.encoder.conv3_x(x)
        self.features.append(x.mean(0))
        x = self.encoder.conv4_x(x)
        self.features.append(x.mean(0))
        x = self.encoder.conv5_x(x)
        self.features.append(x.mean(0))

        return self.features

# a = torch.randn(1, 3, 192, 640).to(device)
# model = SResnetEncoder(pretrained=True, num_input_images=1).to(device)

# b = model(a)
# print(b)