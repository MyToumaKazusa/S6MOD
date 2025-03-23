# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
import numpy as np
from src.models.MamOCL_onpro import MambaNeck_1

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

def M_resnet18(nclasses: int, nf: int = 64, choose: int=0, img_size=32, choose_mamba: str='two projection', num_experts=8, top_k=2) -> nn.Module:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    print("choose mamba:", choose_mamba)
    if choose_mamba == 'two projection':
        return M0_ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf=64, img_size=img_size)
    elif choose_mamba == 'one projection':
        return M1_ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf=64, img_size=img_size)
    elif choose_mamba == 'mamba modulized':
        return M3_ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf=64, img_size=img_size, num_experts=num_experts, top_k=top_k)
        
class M0_ResNet(nn.Module):
    """
    Parallel Projetion! Mamba + Linear
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, img_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(M0_ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        # self.mamba = MambaNeck_1(feat_size=int(img_size/4))
        print("img_size", img_size)
        self.mamba = MambaNeck_1()
        print("init!")
        self.a = self.mamba.a
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.simclr = nn.Linear(nf * 8 * block.expansion, 128)
        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4,
                                       self.mamba
                                       )
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def f_train(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  
        # print("out1:", out.shape) # (B, 64, 32, 32)
        out = self.layer2(out)  
        # print("out2:", out.shape) # (B, 128, 16, 16)
        out = self.layer3(out)  
        # print("out3:", out.shape) # (B, 256, 8, 8)
        out = self.layer4(out)
        # print("out4:", out.shape) # (B, 512, 4, 4)
        # print("out", out.shape)
        out, outputs = self.mamba(out)  #? 
        # print("out_mamba", out.shape)
        # print("out", out.shape)
        
        # (B, D, H, W)
        out = avg_pool2d(out, out.shape[2])  
        # (B, D, 1 ,1)
        out = out.view(out.size(0), -1)  
        # (B, D)

        choose = "flatten"

        if choose == "flatten":
            # plan1 to use the outputs
            # print("dts", outputs['dts'].shape) # (B, 4, 16, 16)
            # print("Bs", outputs['Bs'].shape)
            # print("Cs", outputs['Cs'].shape)
            # outputs['dts'] = outputs['dts'].mean(dim=-1)                  
            outputs['dts'] = outputs['dts'].reshape(outputs['dts'].size(0), -1)
            # outputs['Bs'] = outputs['Bs'].mean(dim=-1)
            outputs['Bs'] = outputs['Bs'].reshape(outputs['Bs'].size(0), -1)
            # outputs['Cs'] = outputs['Cs'].mean(dim=-1)
            outputs['Cs'] = outputs['Cs'].reshape(outputs['Cs'].size(0), -1)
        else:
            "average"
            outputs['dts'] = outputs['dts'].permute(0, 2, 1, 3)
            outputs['dts'] = avg_pool2d(outputs['dts'], outputs['dts'].shape[2])
            outputs['dts'] = outputs['dts'].reshape(outputs['dts'].size(0), -1)
            outputs['Bs'] = outputs['Bs'].permute(0, 2, 1, 3)
            outputs['Bs'] = avg_pool2d(outputs['Bs'], outputs['Bs'].shape[2])
            outputs['Bs'] = outputs['Bs'].reshape(outputs['Bs'].size(0), -1)
            outputs['Cs'] = outputs['Cs'].permute(0, 2, 1, 3)   
            outputs['Cs'] = avg_pool2d(outputs['Cs'], outputs['Cs'].shape[2])
            outputs['Cs'] = outputs['Cs'].reshape(outputs['Cs'].size(0), -1)

        return out, outputs

    def forward(self, x: torch.Tensor, is_simclr=False, is_outputs=False):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out, outputs = self.f_train(x)

        if is_simclr:
        # 会被调用
            feature = out
            out = self.simclr(out) #直接输出特征不投影！！？
            if is_outputs:
                return feature, out, outputs
            return feature, out
        # else:
            # out = self.linear(out)
        if is_outputs:
            return out, outputs
        return out

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out, _ = self.f_train(x)
        out = self.linear(out)
        return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

class M1_ResNet(nn.Module):
    """
    One Projetion! Mamba - The final dimention would be 128
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, img_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(M1_ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.mamba = MambaNeck_1(feat_size=int(img_size/4))
        self.a = self.mamba.a
        self.linear = nn.Linear(nf * 2 * block.expansion, num_classes)
        # self.simclr = nn.Linear(nf * 8 * block.expansion, 128)
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def f_train(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  
        # print("out1:", out.shape) # (B, 64, 32, 32)
        out = self.layer2(out)  
        # print("out2:", out.shape) # (B, 128, 16, 16)
        out = self.layer3(out)  
        # print("out3:", out.shape) # (B, 256, 8, 8)
        out = self.layer4(out)
        # print("out4:", out.shape) # (B, 512, 4, 4)
        return out

    def forward(self, x: torch.Tensor, is_simclr=False, is_outputs=False):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = self.f_train(x) # 4维输出
        
        if is_simclr:
        # 会被调用
            feature = out
            feature = avg_pool2d(feature, feature.shape[2])  
            feature = out.view(feature.size(0), -1)  # （B, 512）

            out, outputs = self.mamba(out)
            out = avg_pool2d(out, out.shape[2])  
            out = out.view(out.size(0), -1)  # (B, 128)

            if is_outputs:
                outputs['dts'] = outputs['dts'].reshape(outputs['dts'].size(0), -1)
                outputs['Bs'] = outputs['Bs'].reshape(outputs['Bs'].size(0), -1)
                outputs['Cs'] = outputs['Cs'].reshape(outputs['Cs'].size(0), -1)
                return feature, out, outputs
            return feature, out
        # else:
            # out = self.linear(out)
        out, outputs = self.mamba(out)
        out = avg_pool2d(out, out.shape[2])  
        out = out.view(out.size(0), -1)  
        if is_outputs:
            outputs['dts'] = outputs['dts'].reshape(outputs['dts'].size(0), -1)
            outputs['Bs'] = outputs['Bs'].reshape(outputs['Bs'].size(0), -1)
            outputs['Cs'] = outputs['Cs'].reshape(outputs['Cs'].size(0), -1)
            return out, outputs
        return out

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = self.f_train(x)
        out, _ = self.mamba(out)
        out = avg_pool2d(out, out.shape[2])  
        out = out.view(out.size(0), -1)  
        # print("out", out.shape)
        out = self.linear(out)
        return out

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

class M3_ResNet(nn.Module):
    """
    Mamba Modulized! Mamba + Linear
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, img_size: int, num_experts: int, top_k: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(M3_ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.mamba = MambaNeck_1(feat_size=int(img_size/4), num_experts=num_experts, top_k=top_k, num_classes=num_classes)
        self.a = self.mamba.a
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.simclr = nn.Linear(nf * 8 * block.expansion, 128)
        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4,
                                       self.mamba
                                       )
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def f_train(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  
        # print("out1:", out.shape) # (B, 64, 32, 32)
        out = self.layer2(out)  
        # print("out2:", out.shape) # (B, 128, 16, 16)
        out = self.layer3(out)  
        # print("out3:", out.shape) # (B, 256, 8, 8)
        out = self.layer4(out)
        # print("out4:", out.shape) # (B, 512, 4, 4)

        return out

    def before_mamba(self, x: torch.Tensor) -> torch.Tensor:
        # return (B, L, D)
        x = self.f_train(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1) # (B, L, D)
        return x

    def after_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f_train(x)
        x = avg_pool2d(x, x.shape[2])
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor, labels=None, is_simclr=False, is_outputs=False, is_original_outputs=False):

        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = self.f_train(x)

        if is_simclr:
            out = avg_pool2d(out, out.shape[2])  
            out = out.view(out.size(0), -1)  
            feature = out
            out = self.simclr(out) #直接输出特征不投影！！？
            return feature, out

        out, outputs = self.mamba(out, labels=labels)  #? 
        # (B, D, H, W)
        out = avg_pool2d(out, out.shape[2])  
        # (B, D, 1 ,1)
        out = out.view(out.size(0), -1)  
        
        if is_outputs:
            outputs['dts'] = outputs['dts'].reshape(outputs['dts'].size(0), -1)
            outputs['Bs'] = outputs['Bs'].reshape(outputs['Bs'].size(0), -1)
            outputs['Cs'] = outputs['Cs'].reshape(outputs['Cs'].size(0), -1)
            # print("outputs_dts!!!!!!!!!", outputs['dts'].shape)
            return out, outputs

        elif is_original_outputs:
            # print("original_outputs_dts", outputs['dts'].shape)
            # print("original_outputs_Bs", outputs['Bs'].shape)
            # print("original_outputs_Cs", outputs['Cs'].shape)
            outputs['dts'] = outputs['dts'].permute(0, 1, 3, 2)
            outputs['dts'] = outputs['dts'].reshape(outputs['dts'].size(0), -1, outputs['dts'].size(3))
            outputs['Bs'] = outputs['Bs'].permute(0, 1, 3, 2)
            outputs['Bs'] = outputs['Bs'].reshape(outputs['Bs'].size(0), -1, outputs['Bs'].size(3))
            outputs['Cs'] = outputs['Cs'].permute(0, 1, 3, 2)
            outputs['Cs'] = outputs['Cs'].reshape(outputs['Cs'].size(0), -1, outputs['Cs'].size(3))
            # print("original_outputs_dts", outputs['dts'].shape)
            # print("original_outputs_Bs", outputs['Bs'].shape)
            # print("original_outputs_Cs", outputs['Cs'].shape)
            return out, outputs

        return out

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = self.f_train(x)
        out = avg_pool2d(out, out.shape[2])  
        out = out.view(out.size(0), -1)  
        out = self.linear(out)
        return out

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def init_weights(model, std=0.01):
    print("Initialize weights of %s with normal dist: mean=0, std=%0.2f" % (type(model), std))
    for m in model.modules():
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 0.1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                m.bias.data.zero_()


class ImageNet_ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ImageNet_ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = nn.Conv2d(3, nf * 1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.simclr = nn.Linear(nf * 8 * block.expansion, 128)
        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def f_train(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  
        out = self.layer2(out)  
        out = self.layer3(out)  
        out = self.layer4(out)  
        out = avg_pool2d(out, out.shape[2])  
        out = out.view(out.size(0), -1)  
        return out

    def forward(self, x: torch.Tensor, is_simclr=False):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = self.f_train(x)

        if is_simclr:
            feature = out
            out = self.simclr(out)
            return feature, out
        else:
            out = self.linear(out)
        return out

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = self.f_train(x)
        out = self.linear(out)
        return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def imagenet_resnet18(nclasses: int, nf: int = 64):
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ImageNet_ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf=64)
