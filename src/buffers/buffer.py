from copyreg import pickle
import torch
import numpy as np
import random as r
import random
import logging as lg
import torch.nn.functional as F
import torch.nn as nn
import random

from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from collections import Counter

from src.utils.data import get_color_distortion
from src.utils.utils import timing, get_device
from src.datasets.memory import MemoryDataset

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomInvert

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

class Buffer(torch.nn.Module):
    def __init__(self, max_size=200, shape=(3,32,32), n_classes=10):
        super().__init__()
        self.n_classes = n_classes  # For print purposes only
        self.max_size = max_size
        self.shape = shape
        self.n_seen_so_far = 0
        self.n_added_so_far = 0
        self.extra_add_so_far = 0
        self.device = get_device()
        if self.shape is not None:
            if len(self.shape) == 3:
                self.register_buffer('buffer_imgs', torch.FloatTensor(self.max_size, self.shape[0], self.shape[1], self.shape[2]).fill_(0))
            elif len(self.shape) == 1:
                self.register_buffer('buffer_imgs', torch.FloatTensor(self.max_size, self.shape[0]).fill_(0))
        self.register_buffer('buffer_labels', torch.LongTensor(self.max_size).fill_(-1))
        # max_size可能导致多余,最好用self.extra_add_so_far作为上界访问
        # self.imgshape = int((self.shape[1] /8)*(self.shape[1] /8))
        # print("self.imgshape", self.imgshape)
        # self.register_buffer('buffer_features', torch.FloatTensor(self.max_size, self.imgshape, 512).fill_(0))
        # self.register_buffer('buffer_dts', torch.FloatTensor(self.max_size, self.imgshape*4, 16).fill_(0))
        # self.register_buffer('buffer_bs', torch.FloatTensor(self.max_size, self.imgshape*4, 16).fill_(0))
        # self.register_buffer('buffer_cs', torch.FloatTensor(self.max_size, self.imgshape*4, 16).fill_(0))

    def get_all(self):
        all_x = self.buffer_imgs[:min(self.current_index, self.max_size)]
        all_y = self.buffer_labels[:min(self.current_index, self.max_size)]
        return all_x, all_y

    def get_all_samples_as_batch(self):
        """
        一次性获取所有样本和标签，作为一个 batch 返回
        :return: (imgs, labels) - 返回全部样本和对应的标签
        """
        # 获取当前已存储的样本数量
        num_samples = self.n_added_so_far

        if num_samples == 0:
            raise ValueError("Buffer is empty. No samples available.")

        # 获取所有存储在 buffer 中的样本和标签
        imgs = self.buffer_imgs[:num_samples].to(self.device)
        labels = self.buffer_labels[:num_samples].to(self.device)

        return imgs, labels

    def get_samples_in_batch(self, num, batch_size):
        # 已获取的样本总数
        retrieved = 0

        while retrieved < num:
            # 每次最多获取 batch_size 数量的样本
            n_imgs = min(batch_size, num - retrieved)
            imgs, labels = self.random_retrieve(n_imgs=n_imgs)

            # 更新已获取的样本总数
            retrieved += n_imgs

            # 使用 yield 逐批返回
            yield imgs, labels

    def save_mamba_params(self, all_features, all_outputs):
        """
        保存训练过程中计算出的特征和 Mamba 参数到 buffer 中
        :param all_features: 存储所有样本的特征 (list of tensors, each with shape (B, D))
        :param all_outputs: 存储所有样本的 Mamba 参数 ('dts', 'Bs', 'Cs')
        """
        self.extra_add_so_far = 0
        num_batches = len(all_features)  # 批次数量

        for batch_idx in range(num_batches):
            # 获取这一批的特征和输出
            batch_features = all_features[batch_idx]  # 形状为 (B, D)
            batch_output = all_outputs[batch_idx]     # 包含 'dts', 'Bs', 'Cs'

            batch_size = batch_features.size(0)  # 当前批次的样本数 (B)

            # 将当前批次的每个样本存储到 buffer 中
            for i in range(batch_size):
                # 保存特征
                self.buffer_features[self.extra_add_so_far] = batch_features[i]

                # 保存 Mamba 参数 ('dts', 'Bs', 'Cs')
                self.buffer_dts[self.extra_add_so_far] = batch_output['dts'][i]
                self.buffer_bs[self.extra_add_so_far] = batch_output['Bs'][i]
                self.buffer_cs[self.extra_add_so_far] = batch_output['Cs'][i]

                # 更新 current_index，确保不超出 buffer 大小
                self.extra_add_so_far += 1
                if self.extra_add_so_far >= self.max_size:
                    break

        # print(" self.buffer_features",  self.buffer_features)
        # print(" self.buffer_dts",  self.buffer_dts)
        # print(" self.buffer_bs",  self.buffer_bs)
        # print(" self.buffer_cs",  self.buffer_cs)

    def get_all_features_and_mamba_params(self):
        """
        从 buffer 中提取所有存储的特征和 Mamba 参数
        :return: 返回所有的特征 (buffer_features) 和 Mamba 参数 (dts, bs, cs)
        """
        # 获取当前已存储的数据量大小，假设 buffer 里有一个 current_index 表示当前位置
        num_samples = self.extra_add_so_far  # 假设 current_index 是当前已存储的数据量

        # 提取 buffer 中的特征和 Mamba 参数
        all_features = self.buffer_features[:num_samples].cuda()  # 提取已存储的部分特征
        all_dts = self.buffer_dts[:num_samples].cuda()           # 提取已存储的 dts
        all_bs = self.buffer_bs[:num_samples].cuda()             # 提取已存储的 Bs
        all_cs = self.buffer_cs[:num_samples].cuda()             # 提取已存储的 Cs

        # 将所有特征和 Mamba 参数作为字典返回
        return {
            'features': all_features,  # 形状为 (num_samples, 16, 512)
            'dts': all_dts,            # 形状为 (num_samples, 16, 512) 或 (num_samples, 16, 16)
            'bs': all_bs,              # 形状为 (num_samples, 16, 512) 或 (num_samples, 16, 16)
            'cs': all_cs               # 形状为 (num_samples, 16, 512) 或 (num_samples, 16, 16)
        }

    def update(self, imgs, labels=None):
        raise NotImplementedError

    def stack_data(self, img, label):
        if self.n_seen_so_far < self.max_size:
            self.buffer_imgs[self.n_seen_so_far] = img
            self.buffer_labels[self.n_seen_so_far] = label
            self.n_added_so_far += 1

    def replace_data(self, idx, img, label):
        self.buffer_imgs[idx] = img
        self.buffer_labels[idx] = label
        self.n_added_so_far += 1
        # print("self.n_added_so_far: ", self.n_added_so_far)
    
    def is_empty(self):
        return self.n_added_so_far == 0
    
    def random_retrieve(self, n_imgs=100, present=None):
        if self.n_added_so_far < n_imgs:
            lg.debug(f"""Cannot retrieve the number of requested images from memory {self.n_added_so_far}/{n_imgs}""")
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far]
        
        # print("random")

        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        # self.count_tasks(ret_labels, present)
        
        return ret_imgs, ret_labels
    
    def only_retrieve(self, n_imgs, desired_labels):
        """Retrieve images belonging only to the set of desired labels

        Args:
            n_imgs (int):                    Number of images to retrieve 
            desired_labels (torch.Tensor): tensor of desired labels to retrieve from
        """
        desired_labels = torch.tensor(desired_labels)
        valid_indexes = torch.isin(self.buffer_labels, desired_labels).nonzero().view(-1)
        n_out = min(n_imgs, len(valid_indexes))
        out_indexes = np.random.choice(valid_indexes, n_out)
        
        return self.buffer_imgs[out_indexes], self.buffer_labels[out_indexes]
    
    def except_retrieve(self, n_imgs, undesired_labels):
        """Retrieve images except images of undesired labels

        Args:
            n_imgs (int):                  Number of images to retrieve 
            desired_labels (torch.Tensor): tensor of desired labels to retrieve from
        """
        undesired_labels = torch.tensor(undesired_labels + [-1])
        valid_indexes = (~torch.isin(self.buffer_labels, undesired_labels)).nonzero().view(-1)
        n_out = min(n_imgs, len(valid_indexes))
        out_indexes = np.random.choice(valid_indexes, n_out)
        
        return self.buffer_imgs[out_indexes], self.buffer_labels[out_indexes]
    
    def dist_retrieve(self, means, model, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            lg.debug(f"""Cannot retrieve the number of requested images from memory {self.n_added_so_far}/{n_imgs}""")
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far]
        
        # model.eval()
        with torch.no_grad():
            _, p_mem = model(self.buffer_imgs[:self.n_added_so_far].to(self.device))

        m = torch.zeros((p_mem.shape[1], self.n_classes)).to(self.device)
        for c in means:
            m[:, int(float(c))] = means[f'{c}']

        dists = p_mem @ m
        # Get distances from kown classes only
        dists = dists[torch.arange(dists.size(0)), self.buffer_labels[:self.n_added_so_far]]
        sorted_idx = dists.sort(descending=True).indices
        ret_indexes = []
        # ensuring we get some of each class
        for c in self.buffer_labels[:self.n_added_so_far].unique():
            idx = torch.where((self.buffer_labels[:self.n_added_so_far][sorted_idx] == c))[0][:int(n_imgs/len(self.buffer_labels[:self.n_added_so_far].unique()))]
            ret_indexes.append(idx)
        ret_indexes = torch.cat(ret_indexes)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]

        return ret_imgs, ret_labels
    
    @staticmethod
    def empty_transform(x):
        return x

    def count_tasks(self, final_labels, present):
        """
        统计final_labels中在present中的属于task2，否则属于task1的标签数量。

        参数:
        - final_labels: 需要处理的标签列表 (list 或 numpy array)
        - present: 已知标签的Tensor

        返回:
        - task1_count: 不在 present 中的标签的数量
        - task2_count: 在 present 中的标签的数量
        """
        task1_count = 0  # 不在 present 中的标签计数
        task2_count = 0  # 在 present 中的标签计数
        
        # 遍历 final_labels 并统计 task1 和 task2 的数量
        for label in final_labels:
            if label in present:  # 如果标签在 present 中
                task2_count += 1
            else:
                task1_count += 1
        
        print(f"task1_count: {task1_count}, task2_count: {task2_count}")
        # return task1_count, task2_count


    def min_mamba_retrieve(self, model, n_imgs=100, mul_num=2, transform=None, present=None):
        if self.n_added_so_far < mul_num * n_imgs:
            lg.debug(f"""Cannot retrieve the number of requested images from memory {self.n_added_so_far}/{mul_num * n_imgs}""")
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far]
        
        mul_num = min(mul_num, self.max_size//n_imgs)

        # 从 buffer 中随机抽取 mul_num * n_imgs 个样本
        # print("self.n_added_so_far: ", self.n_added_so_far)
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), mul_num * n_imgs)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        
        # 将样本按每 mul_num 个一组进行分组
        selected_indexes = []
        for i in range(n_imgs):
            group_imgs = ret_imgs[i * mul_num: (i + 1) * mul_num]
            group_labels = ret_labels[i * mul_num: (i + 1) * mul_num]
            
            model.eval()
            with torch.no_grad():
                _, group_outputs = model(transform(group_imgs.to(self.device)), is_original_outputs=True)
            
            uncertainties = torch.zeros(len(group_imgs)).to(self.device)
            for param in range(3):
                if param == 0:
                    o_param = 'dts'
                elif param == 1:
                    o_param = 'Bs'
                else:
                    o_param = 'Cs'
                outputs = group_outputs[o_param]
                # 计算每个样本的熵，并选择熵最大的样本
                uncertainties += self.calculate_uncertainty_entropy(outputs)

            _, max_uncertainty_idx = uncertainties.max(dim=0)
            selected_indexes.append(i * mul_num + max_uncertainty_idx)
        # print("selected_indexes: ", selected_indexes)
        
        # 返回保留的 n_imgs 个样本
        final_imgs = ret_imgs[selected_indexes]
        final_labels = ret_labels[selected_indexes]

        self.count_tasks(final_labels, present)
        
        return final_imgs, final_labels

    def mamba_retrieve(self, model, n_imgs=100, mul_num=2, transform=None, present=None):
        if self.n_added_so_far < mul_num * n_imgs:
            lg.debug(f"""Cannot retrieve the number of requested images from memory {self.n_added_so_far}/{mul_num * n_imgs}""")
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far]
        
        mul_num = min(mul_num, self.max_size//n_imgs)

        # 从 buffer 中随机抽取 mul_num * n_imgs 个样本
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), mul_num * n_imgs)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        
        # 将样本按每 mul_num 个一组进行分组
        selected_indexes = []
        for i in range(n_imgs):
            group_imgs = ret_imgs[i * mul_num: (i + 1) * mul_num]
            group_labels = ret_labels[i * mul_num: (i + 1) * mul_num]
            
            model.eval()
            with torch.no_grad():
                _, group_outputs = model(transform(group_imgs.to(self.device)), is_original_outputs=True)
            
            uncertainties = torch.zeros(len(group_imgs)).to(self.device)
            for param in range(3):
                if param == 0:
                    o_param = 'dts'
                elif param == 1:
                    o_param = 'Bs'
                else:
                    o_param = 'Cs'
                outputs = group_outputs[o_param]
                # 计算每个样本的熵，并选择熵最大的样本
                uncertainties += self.calculate_uncertainty_entropy(outputs)

            _, max_uncertainty_idx = uncertainties.min(dim=0)
            selected_indexes.append(i * mul_num + max_uncertainty_idx.item())
        
        # 返回保留的 n_imgs 个样本
        final_imgs = ret_imgs[selected_indexes]
        # print("final_imgs", final_imgs)
        final_labels = ret_labels[selected_indexes]
        # print("selected_indexes", selected_indexes)
        # print("final_labels", final_labels)
        # print("selected_indexes.item()", selected_indexes.item())
        # self.count_tasks(final_labels, present)
        # 统计labels的分布
        
        return final_imgs, final_labels

    def dist_update(self, means, model, imgs, labels, **kwargs):
        # model.eval()
        # with torch.no_grad():
        #     _, p_mem = model(self.buffer_imgs[:self.n_added_so_far].to(self.device))
        
        # m = torch.zeros((p_mem.shape[1], self.n_classes)).to(self.device)
        # for c in means:
        #     m[:, int(float(c))] = means[f'{c}']

        # dists = p_mem @ m
        for stream_data, stream_label in zip(imgs, labels):
            if self.n_added_so_far < self.max_size:
                self.stack_data(stream_data, stream_label)
            else:
                max_img_per_class = self.get_max_img_per_class()
                class_indexes = self.get_indexes_of_class(stream_label)
                # Do nothing if class has reached maximum number of images
                if len(class_indexes) <= max_img_per_class:
                    # Drop img of major class if not
                    major_class = self.get_major_class()
                    class_indexes = self.get_indexes_of_class(major_class)

                    # compute distances to mean
                    model.eval()
                    with torch.no_grad():
                        _, p_mem = model(self.buffer_imgs[class_indexes.squeeze()].to(self.device))
                    
                    m = means[f'{major_class}.0'].to(self.device)

                    dists = p_mem @ m
                    # idx = class_indexes.squeeze()[dists.argmax()]
                    idx = class_indexes.squeeze()[dists.argmin()]
                    self.replace_data(idx, stream_data, stream_label)
            self.n_seen_so_far += 1
    
    def bootstrap_retrieve(self, n_imgs=100):
        if self.n_added_so_far == 0:
            return torch.Tensor(), torch.Tensor() 
        ret_indexes = [r.randint(0, min(self.n_added_so_far, self.max_size)-1) for _ in range(n_imgs)]            
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]

        return ret_imgs, ret_labels
        
    def n_data(self):
        return len(self.buffer_labels[self.buffer_labels >= 0])

    def get_all(self):
        return self.buffer_imgs[:min(self.n_added_so_far, self.max_size)],\
             self.buffer_labels[:min(self.n_added_so_far, self.max_size)]

    def get_indexes_of_class(self, label):
        return torch.nonzero(self.buffer_labels == label)
    
    def get_indexes_out_of_class(self, label):
        return torch.nonzero(self.buffer_labels != label)

    def is_full(self):
        return self.n_data() == self.max_size

    def get_labels_distribution(self):
        np_labels = self.buffer_labels.numpy().astype(int)
        counts = np.bincount(np_labels[self.buffer_labels >= 0], minlength=self.n_classes)
        tot_labels = len(self.buffer_labels[self.buffer_labels >= 0])
        if tot_labels > 0:
            return counts / len(self.buffer_labels[self.buffer_labels >= 0])
        else:
            return counts

    def get_major_class(self):
        np_labels = self.buffer_labels.numpy().astype(int)
        counts = np.bincount(np_labels[self.buffer_labels >= 0])
        return counts.argmax()

    def get_max_img_per_class(self):
        n_classes_in_memory = len(self.buffer_labels.unique())
        return int(len(self.buffer_labels[self.buffer_labels >= 0]) / n_classes_in_memory) # 有效样本/类别数

    def my_get_max_img_per_class(self):
        n_classes_in_memory = len(self.buffer_labels.unique())
        return int(self.max_size / n_classes_in_memory) # buffer大小/类别数

    import torch.nn.functional as F

    def batch_mamba_update(self, means, model, imgs, labels, transform, **kwargs):
        # outputs
        for stream_data, stream_label in zip(imgs, labels):
            stream_data = stream_data.cpu()
            stream_label = stream_label.cpu()
            if self.n_added_so_far < self.max_size:
                self.stack_data(stream_data, stream_label)
            else:
                max_img_per_class = self.my_get_max_img_per_class()
                class_indexes = self.get_indexes_of_class(stream_label)
                
                # Do nothing if class has reached maximum number of images
                # 如果当前类别的样本数小于允许的每个类别最大样本数
                if len(class_indexes) < max_img_per_class:
                    # Drop img of major class if not
                    major_class = self.get_major_class()
                    class_indexes = self.get_indexes_of_class(major_class)

                    # compute cosine similarity to mean for multiple parameters
                    model.eval()
                    with torch.no_grad():
                        # Get the model outputs dictionary
                        _, outputs = model(transform(self.buffer_imgs[class_indexes.squeeze()].to(self.device)), is_outputs=True)

                    summed_cosine_sim = torch.zeros(len(class_indexes)).to(self.device)

                    for param in range(3):
                        # Retrieve the mean vector for the current parameter and class
                        mean_param = means[major_class][param].to(self.device)
                        # Compute cosine similarity between outputs and mean
                        if param == 0:
                            o_param = 'dts'
                            # continue
                            print("use dts!!!")
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                            print("use Bs!!!")
                        else:
                            o_param = 'Cs'
                            # continue
                            print("use Cs!!!")
                        cosine_sim = F.cosine_similarity(outputs[o_param], mean_param.unsqueeze(0), dim=1)
                        # Add the cosine similarity to the summed result
                        summed_cosine_sim += cosine_sim

                    # Find the index with the maximum summed cosine similarity
                    idx = class_indexes.squeeze()[summed_cosine_sim.argmax()]

                    # Replace the selected data in the buffer
                    self.replace_data(idx, stream_data, stream_label)
                # 如果当前类别的样本数大于等于允许的每个类别最大样本数
                else:
                    # 当前类别已达到最大样本数，需判断是否替换缓冲区中的样本
                    # 获取当前类别在缓冲区中的所有索引
                    class_indexes = self.get_indexes_of_class(stream_label)

                    model.eval()
                    with torch.no_grad():
                        # 获取新样本的模型输出
                        _, new_output = model(transform(stream_data.to(self.device).unsqueeze(0)), is_outputs=True)
                        # 获取缓冲区中当前类别所有样本的模型输出
                        _, buffer_outputs = model(transform(self.buffer_imgs[class_indexes.squeeze()].to(self.device)), is_outputs=True)

                    # 计算新样本与均值的累加余弦相似度
                    new_summed_cosine_sim = torch.zeros(1).to(self.device)
                    for param in range(3):
                        # print("means[stream_label]: ", means[stream_label.item()])
                        mean_param = means[stream_label.item()][param].to(self.device)
                        if param == 0:
                            o_param = 'dts'
                            # continue
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                        else:
                            o_param = 'Cs'
                            # continue
                        cosine_sim = F.cosine_similarity(new_output[o_param].squeeze(), mean_param, dim=0)
                        new_summed_cosine_sim += cosine_sim

                    # 计算缓冲区中所有样本与均值的累加余弦相似度
                    summed_cosine_sim_buffer = torch.zeros(len(class_indexes)).to(self.device)
                    for param in range(3):
                        mean_param = means[stream_label.item()][param].to(self.device)
                        if param == 0:
                            o_param = 'dts'
                            # continue
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                        else:
                            o_param = 'Cs'
                            # continue
                        cosine_sim = F.cosine_similarity(buffer_outputs[o_param], mean_param.unsqueeze(0), dim=1)
                        summed_cosine_sim_buffer += cosine_sim

                    # 找到缓冲区中累加余弦相似度最小的样本
                    max_sim, max_idx = summed_cosine_sim_buffer.max(dim=0)

                    # 如果新样本的相似度小于缓冲区中最大的相似度，则进行替换
                    if new_summed_cosine_sim.item() < max_sim.item():
                        # 获取需要替换的样本在缓冲区中的实际索引
                        idx_to_replace = class_indexes[max_idx].item()
                        # 替换缓冲区中的样本
                        self.replace_data(idx_to_replace, stream_data, stream_label)
                    # 否则，不进行任何操作
                    
            # Increment the seen data counter
            self.n_seen_so_far += 1

    def buffer_mamba_update(self, model, imgs, labels, transform, similarity_threshold=2.9, **kwargs):
        for stream_data, stream_label in zip(imgs, labels):
            stream_data = stream_data.cpu()
            stream_label = stream_label.cpu()
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))

            if self.n_added_so_far < self.max_size:
                self.stack_data(stream_data, stream_label)
            else:
                max_img_per_class = self.my_get_max_img_per_class()
                class_indexes = self.get_indexes_of_class(stream_label)

                # 感觉这里有问题！！！
                if len(class_indexes) < max_img_per_class:
                    model.eval()
                    # 随机取代buffer中的样本
                    reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
                    if self.n_seen_so_far < self.max_size:
                        reservoir_idx = self.n_added_so_far
                    if reservoir_idx < self.max_size:
                            self.replace_data(reservoir_idx, stream_data, stream_label)
                else:
                    # Similar logic as above for replacing samples if the current class is full
                    class_indexes = self.get_indexes_of_class(stream_label)
                    model.eval()

                    # reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
                    # if self.n_seen_so_far < self.max_size:
                    #     reservoir_idx = self.n_added_so_far
                    # if reservoir_idx < self.max_size:
                    #         self.replace_data(reservoir_idx, stream_data, stream_label)
                    with torch.no_grad():
                        _, new_output = model(transform(stream_data.to(self.device).unsqueeze(0)), is_outputs=True)
                        _, buffer_outputs = model(transform(self.buffer_imgs[class_indexes.squeeze()].to(self.device)), is_outputs=True)

                    new_cosine_sim = []
                    for param in range(3):
                        o_param = ['dts', 'Bs', 'Cs'][param]
                        sim = F.cosine_similarity(new_output[o_param].squeeze().detach(), buffer_outputs[o_param].detach(), dim=1)
                        # print("sim: ", sim)
                        new_cosine_sim.append(sim)

                    # print("new_cosine_sim: ", new_cosine_sim)
                    summed_cosine_sim_buffer = sum(new_cosine_sim) # [N]
                    # print("summed_cosine_sim_buffer: ", summed_cosine_sim_buffer)

                    if (summed_cosine_sim_buffer > similarity_threshold).any():
                        continue
                    else:
                        max_sim, max_idx = summed_cosine_sim_buffer.max(dim=0)
                        idx_to_replace = class_indexes[max_idx].item()
                        self.replace_data(idx_to_replace, stream_data, stream_label)

            self.n_seen_so_far += 1

    def buffer1_mamba_update(self, model, imgs, labels, transform, similarity_threshold=0.98, **kwargs):
        for stream_data, stream_label in zip(imgs, labels):
            stream_data = stream_data.cpu()
            stream_label = stream_label.cpu()
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))

            if self.n_added_so_far < self.max_size:
                self.stack_data(stream_data, stream_label)
            else:
                max_img_per_class = self.my_get_max_img_per_class()
                class_indexes = self.get_indexes_of_class(stream_label)

                # 感觉这里有问题！！！
                if len(class_indexes) < max_img_per_class:
                    model.eval()
                    # 随机取代buffer中的样本
                    reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
                    if self.n_seen_so_far < self.max_size:
                        reservoir_idx = self.n_added_so_far
                    if reservoir_idx < self.max_size:
                            self.replace_data(reservoir_idx, stream_data, stream_label)
                else:
                    # Similar logic as above for replacing samples if the current class is full
                    class_indexes = self.get_indexes_of_class(stream_label)
                    model.eval()

                    # reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
                    # if self.n_seen_so_far < self.max_size:
                    #     reservoir_idx = self.n_added_so_far
                    # if reservoir_idx < self.max_size:
                    #         self.replace_data(reservoir_idx, stream_data, stream_label)
                    with torch.no_grad():
                        new_feature = model(transform(stream_data.to(self.device).unsqueeze(0)))
                        buffer_features = model(transform(self.buffer_imgs[class_indexes.squeeze()].to(self.device)))

                    # print("new_feature: ", new_feature.shape)
                    # print("buffer_features: ", buffer_features.shape)

                    new_cosine_sim = []
                    sim = F.cosine_similarity(new_feature.squeeze().detach(), buffer_features.detach(), dim=1)
                    # print("sim: ", sim)
                    new_cosine_sim.append(sim)

                    # print("new_cosine_sim: ", new_cosine_sim)
                    summed_cosine_sim_buffer = sum(new_cosine_sim) # [N]
                    # print("summed_cosine_sim_buffer: ", summed_cosine_sim_buffer)

                    if (summed_cosine_sim_buffer > similarity_threshold).any():
                        continue
                    else:
                        max_sim, max_idx = summed_cosine_sim_buffer.max(dim=0)
                        idx_to_replace = class_indexes[max_idx].item()
                        self.replace_data(idx_to_replace, stream_data, stream_label)

            self.n_seen_so_far += 1

    def mamba_update(self, type, means, model, imgs, labels, transform, **kwargs):
        if type == 'max':
            self.max_mamba_update(means, model, imgs, labels, transform, **kwargs)
        elif type == 'batch':
            self.batch_mamba_update(means, model, imgs, labels, transform, **kwargs)
        elif type == 'buffer':
            self.buffer_mamba_update(model, imgs, labels, transform)
        elif type == 'min':
            self.min_mamba_update(means, model, imgs, labels, transform, **kwargs)
        elif type == 'mid':
            self.mid_mamba_update(means, model, imgs, labels, transform, **kwargs)
        elif type == 'onlymid':
            label = random.randint(0, 1)
            if label == 0:
                self.min_mamba_update(means, model, imgs, labels, transform, **kwargs)
            else:
                self.max_mamba_update(means, model, imgs, labels, transform, **kwargs)
        else:
            raise NotImplementedError(f"Memory update type {type} not implemented")

    def calculate_uncertainty_var(self, output):
        # ???为什么dts的特征维度只有16
        # input: [B, L, D]
        # output: [B]

        # print("output: ", output.shape)
        # Step 1: 计算每个样本在时间步上的均值，结果形状为 [B, D]
        mean_per_time_step = output.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        # Step 2: 计算时间步的方差
        # 对每个样本的每个特征计算时间步之间的方差
        variance_per_time_step = ((output - mean_per_time_step) ** 2).mean(dim=1)  # [B, D]

        # Step 3: 对每个特征的方差进行求和，得到每个样本的总方差，形状为 [B]
        total_variance = variance_per_time_step.sum(dim=-1)  # [B]

        return total_variance

    def calculate_uncertainty_del(self, output):
        """
        计算基于 Mamba 模型参数 Δ, b, c 的变化率不确定性
        Args:
            output_delta (torch.Tensor): 模型输出的 Δ 参数, 形状 [B, L, D]
            output_b (torch.Tensor): 模型输出的 b 参数, 形状 [B, L, D]
            output_c (torch.Tensor): 模型输出的 c 参数, 形状 [B, L, D]
        
        Returns:
            torch.Tensor: 每个样本的不确定性度量, 形状 [B]
        """

        # Step 1: 计算 Δ 参数的变化率 (即相邻时间步之间的差值)
        delta_rate_of_change = torch.abs(output[:, 1:, :] - output[:, :-1, :])  # [B, L-1, D]
        delta_uncertainty = delta_rate_of_change.mean(dim=1).sum(dim=-1)  # [B]

        return delta_uncertainty

    def calculate_uncertainty_entropy(self, output):
        # Step 1: 计算 Δ 参数的变化率 (即相邻时间步之间的差值)
        rate_of_change = torch.abs(output[:, 1:, :] - output[:, :-1, :])  # [B, L-1, D]
        # Step 1: 对变化率进行归一化
        rate_of_change = rate_of_change / (rate_of_change.sum(dim=1, keepdim=True) + 1e-9)  # 避免除零问题
            
        # Step 2: 计算熵，公式为 -Σ(p * log(p))，其中 p 是变化率的分布
        entropy = - (rate_of_change * torch.log(rate_of_change + 1e-9)).sum(dim=1).sum(dim=-1)  # [B]
        # print("entropy: ", entropy)

        return entropy

    def mid_mamba_update(self, means, model, imgs, labels, transform, **kwargs):
        # random mid + uncertainty
        for stream_data, stream_label in zip(imgs, labels):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1)) # [0, self.n_seen_so_far]
            if self.n_seen_so_far < self.max_size:
                # 如果当前没满，直接加入最后一个位置
                reservoir_idx = self.n_added_so_far
                self.replace_data(reservoir_idx, stream_data, stream_label)
            elif reservoir_idx < self.max_size:
                # 计算待替换样本类别中所有样本的不确定性
                buffer_label = self.buffer_labels[reservoir_idx]
                buffer_indexes = self.get_indexes_of_class(buffer_label)

                model.eval()
                with torch.no_grad():
                    _, buffer_outputs = model(transform(self.buffer_imgs[buffer_indexes.squeeze()].to(self.device)), is_original_outputs=True)
                
                uncertainty_buffer = torch.zeros(len(buffer_indexes)).to(self.device)
                for param in range(3):
                    o_param = 'dts' if param == 0 else ('Bs' if param == 1 else 'Cs')
                    uncertainty_buffer += self.calculate_uncertainty_entropy(buffer_outputs[o_param])

                # 对不确定性排序并选取中间的索引
                sorted_uncertainties, sorted_indices = torch.sort(uncertainty_buffer)
                mid_idx = sorted_indices[len(sorted_indices) // 2]  # 选取中间的不确定性样本
                reservoir_idx = buffer_indexes[mid_idx].item()
                self.replace_data(reservoir_idx, stream_data, stream_label)

            self.n_seen_so_far += 1


    def min_mamba_update(self, means, model, imgs, labels, transform, **kwargs):
        # 如果有效可以把dist关了
        # random min + uncertainty
        for stream_data, stream_label in zip(imgs, labels):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1)) # [0, self.n_seen_so_far]
            if self.n_seen_so_far < self.max_size:
                # 如果当前没满，直接加入最后一个位置
                reservoir_idx = self.n_added_so_far
                self.replace_data(reservoir_idx, stream_data, stream_label)
            # 如果还有位置并且random的数值在容量中，判断后进行替换
            elif reservoir_idx < self.max_size:
                # 计算待替换样本类别中最大的相似度
                buffer_label = self.buffer_labels[reservoir_idx]
                buffer_indexes = self.get_indexes_of_class(buffer_label)

                model.eval()
                with torch.no_grad():
                    # 原来是(B, 1)需要变成(B)
                    # print("buffer_indexes: ", buffer_indexes.shape)
                    # print("buffer_indexes.squeeze()", buffer_indexes.squeeze().shape)
                    _, buffer_outputs = model(transform(self.buffer_imgs[buffer_indexes.squeeze()].to(self.device)), is_original_outputs=True)
                # 计算缓冲区中所有样本与均值的不确定性
                uncertainty_buffer = torch.zeros(len(buffer_indexes)).to(self.device)
                for param in range(3):
                    # print("param:", param)
                    if param == 0:
                        o_param = 'dts'
                        # continue
                    elif param == 1:
                        o_param = 'Bs'
                        # continue
                    else:
                        o_param = 'Cs'
                        # continue
                    uncertainty_buffer += self.calculate_uncertainty_entropy(buffer_outputs[o_param])  
                    # print("buffer_outputs[o_param]", buffer_outputs[o_param].shape)
                    # print("uncertainty_buffer: ", uncertainty_buffer.shape)
                    # print("uncertainty_buffer: ", uncertainty_buffer)

                min_uncer, min_idx = uncertainty_buffer.min(dim=0)

                if buffer_label == stream_label:
                    # 这里真一样吗？
                    # print("stream_label= ", stream_label)
                    model.eval()
                    with torch.no_grad():
                        _, new_output = model(transform(stream_data.to(self.device).unsqueeze(0)), is_original_outputs=True)
                    uncertainty_new = 0.0
                    for param in range(3):
                        if param == 0:
                            o_param = 'dts'
                            # continue
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                        else:
                            o_param = 'Cs'
                            # continue
                        # 这里不该加squeeze！！！
                        uncertainty_new += self.calculate_uncertainty_entropy(new_output[o_param])
                    # print("uncertainty_new: ", uncertainty_new)
                    # print("min_uncer: ", min_uncer)
                    # 如果当前样本的不确定性更小，就不进行替换了
                    if uncertainty_new.item() < min_uncer.item():
                        self.n_seen_so_far += 1
                        continue
            
                reservoir_idx = buffer_indexes[min_idx].item()
                self.replace_data(reservoir_idx, stream_data, stream_label)

            # Increment the seen data counter
            self.n_seen_so_far += 1
            

    def max_mamba_update(self, means, model, imgs, labels, transform, **kwargs):
        # 如果有效可以把dist关了
        # random min + uncertainty
        for stream_data, stream_label in zip(imgs, labels):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1)) # [0, self.n_seen_so_far]
            if self.n_seen_so_far < self.max_size:
                # 如果当前没满，直接加入最后一个位置
                reservoir_idx = self.n_added_so_far
                self.replace_data(reservoir_idx, stream_data, stream_label)
            # 如果还有位置并且random的数值在容量中，判断后进行替换
            elif reservoir_idx < self.max_size:

                # 计算待替换样本类别中最大的相似度
                buffer_index = reservoir_idx
                buffer_label = self.buffer_labels[buffer_index]
                buffer_indexes = self.get_indexes_of_class(buffer_label)

                model.eval()
                with torch.no_grad():
                    _, buffer_outputs = model(transform(self.buffer_imgs[buffer_indexes.squeeze()].to(self.device)), is_original_outputs=True)
                # 计算缓冲区中所有样本与均值的不确定性
                uncertainty_buffer = torch.zeros(len(buffer_indexes)).to(self.device)
                for param in range(3):
                    # print("param:", param)
                    if param == 0:
                        o_param = 'dts'
                        # continue
                    elif param == 1:
                        o_param = 'Bs'
                        # continue
                    else:
                        o_param = 'Cs'
                        # continue
                    uncertainty_buffer = self.calculate_uncertainty_entropy(buffer_outputs[o_param])
                    # print("uncertainty_buffer: ", uncertainty_buffer)

                # 找到缓冲区中不确定性最小的样本
                min_uncer, min_idx = uncertainty_buffer.max(dim=0)

                if buffer_label == stream_label:
                    # print("stream_label= ", stream_label)
                    model.eval()
                    with torch.no_grad():
                        _, new_output = model(transform(stream_data.to(self.device).unsqueeze(0)), is_original_outputs=True)
                    for param in range(3):
                        if param == 0:
                            o_param = 'dts'
                            # continue
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                        else:
                            o_param = 'Cs'
                            # continue
                        # 这里不该加squeeze！！！
                        uncertainty_new = self.calculate_uncertainty_entropy(new_output[o_param])
                        # print("uncertainty_new: ", uncertainty_new)
                        # print("uncertainty_buffer: ", uncertainty_buffer)
                    # 如果当前样本的相似度更小，就不进行替换了
                    if uncertainty_new.item() > min_uncer.item():
                        print("Yes!!!")
                        self.n_seen_so_far += 1
                        continue
            
                reservoir_idx = buffer_indexes[min_idx].item()
                self.replace_data(reservoir_idx, stream_data, stream_label)

            # Increment the seen data counter
            self.n_seen_so_far += 1

    def max2_mamba_update(self, means, model, imgs, labels, transform, **kwargs):
        # random max pro
        # 相比于前者加上了同类别判断的支路
        # 是否eval？？？
        # 以random为主体，判断当前样本与抽取样本分别在对应类之间相似度的大小，保留更小的
        for stream_data, stream_label in zip(imgs, labels):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1)) # [0, self.n_seen_so_far]
            if self.n_seen_so_far < self.max_size:
                # 如果当前没满，直接加入最后一个位置
                reservoir_idx = self.n_added_so_far
                self.replace_data(reservoir_idx, stream_data, stream_label)
            # 如果还有位置并且random的数值在容量中，判断后进行替换
            elif reservoir_idx < self.max_size:
                # model.eval()?
                # 计算待替换样本类别中最大的相似度
                buffer_index = reservoir_idx
                # 这里的索引可能不对，需要检查！！！？？？
                buffer_label = self.buffer_labels[buffer_index]
                buffer_indexes = self.get_indexes_of_class(buffer_label)

                model.eval()
                with torch.no_grad():
                    _, buffer_outputs = model(transform(self.buffer_imgs[buffer_indexes.squeeze()].to(self.device)), is_outputs=True)

                # 计算缓冲区中所有样本与均值的累加余弦相似度
                summed_cosine_sim_buffer = torch.zeros(len(buffer_indexes)).to(self.device)
                for param in range(3):
                    mean_param = means[buffer_label.item()][param].to(self.device)
                    if param == 0:
                        o_param = 'dts'
                        # continue
                    elif param == 1:
                        o_param = 'Bs'
                        # continue
                    else:
                        o_param = 'Cs'
                        # continue
                    cosine_sim = F.cosine_similarity(buffer_outputs[o_param], mean_param.unsqueeze(0), dim=1)
                    summed_cosine_sim_buffer += cosine_sim

                # 找到缓冲区中累加余弦相似度最大的样本
                max_sim, max_idx = summed_cosine_sim_buffer.max(dim=0)
                reservoir_idx = buffer_indexes[max_idx].item()
                self.replace_data(reservoir_idx, stream_data, stream_label)
            # Increment the seen data counter
            self.n_seen_so_far += 1

    def max3_mamba_update(self, means, model, imgs, labels, transform, **kwargs):
        # random max + similarity
        # print之后发现不同列表，b的区分度比较高
        # random dist的数值多次测试
        # 最后用相似度判断是否replace的方法不可行，因为新样本的相似度太大，比不过旧样本经过筛选后更小的相似度
        # 是否需要判断random到的样本类别，相同则需要做额外计算 （如果现有方法提升，可能加上后还能提升）
        # 是否eval？？？
        # 以random为主体，判断当前样本与抽取样本分别在对应类之间相似度的大小，保留更小的
        # 将本类别的判断也加进去
        for stream_data, stream_label in zip(imgs, labels):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1)) # [0, self.n_seen_so_far]
            if self.n_seen_so_far < self.max_size:
                # 如果当前没满，直接加入最后一个位置
                reservoir_idx = self.n_added_so_far
                self.replace_data(reservoir_idx, stream_data, stream_label)
            # 如果还有位置并且random的数值在容量中，判断后进行替换
            elif reservoir_idx < self.max_size:
                # 计算待替换样本类别中最大的相似度
                buffer_index = reservoir_idx
                # 这里的索引可能不对，需要检查！！！？？？
                buffer_label = self.buffer_labels[buffer_index]
                buffer_indexes = self.get_indexes_of_class(buffer_label)

                # 计算buffer样本的相似度顺序
                model.eval()
                with torch.no_grad():
                    _, buffer_outputs = model(transform(self.buffer_imgs[buffer_indexes.squeeze()].to(self.device)), is_outputs=True)

                # 计算缓冲区中所有样本与均值的累加余弦相似度
                summed_cosine_sim_buffer = torch.zeros(len(buffer_indexes)).to(self.device)
                for param in range(3):
                    # print("param:", param)
                    mean_param = means[buffer_label.item()][param].to(self.device)
                    if param == 0:
                        o_param = 'dts'
                        continue
                    elif param == 1:
                        o_param = 'Bs'
                        # continue
                    else:
                        o_param = 'Cs'
                        continue
                    
                    cosine_sim = F.cosine_similarity(buffer_outputs[o_param], mean_param.unsqueeze(0), dim=1)
                    summed_cosine_sim_buffer += cosine_sim

                # 找到缓冲区中累加余弦相似度最大的样本
                max_sim, max_idx = summed_cosine_sim_buffer.max(dim=0)

                if buffer_label == stream_label:
                    model.eval()
                    with torch.no_grad():
                        _, new_output = model(transform(stream_data.to(self.device).unsqueeze(0)), is_outputs=True)
                    summed_cosine_sim_new = torch.zeros(1).to(self.device)
                    for param in range(3):
                        mean_param = means[stream_label.item()][param].to(self.device)
                        if param == 0:
                            o_param = 'dts'
                            continue
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                        else:
                            o_param = 'Cs'
                            continue

                        cosine_sim = F.cosine_similarity(new_output[o_param].squeeze(), mean_param, dim=0)
                        summed_cosine_sim_new += cosine_sim
                    # 如果当前样本的相似度更大，就不进行替换了
                    if summed_cosine_sim_new.item() > max_sim.item():
                        self.n_seen_so_far += 1
                        continue
            
                reservoir_idx = buffer_indexes[max_idx].item()
                self.replace_data(reservoir_idx, stream_data, stream_label)

            # Increment the seen data counter
            self.n_seen_so_far += 1

    def min2_mamba_update(self, means, model, imgs, labels, transform, **kwargs):
        # random min
        # print之后发现不同列表，b的区分度比较高
        # random dist的数值多次测试
        # 最后用相似度判断是否replace的方法不可行，因为新样本的相似度太大，比不过旧样本经过筛选后更小的相似度
        # 是否需要判断random到的样本类别，相同则需要做额外计算 （如果现有方法提升，可能加上后还能提升）
        # 是否eval？？？
        # 以random为主体，判断当前样本与抽取样本分别在对应类之间相似度的大小，保留更小的
        # 将本类别的判断也加进去
        for stream_data, stream_label in zip(imgs, labels):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1)) # [0, self.n_seen_so_far]
            if self.n_seen_so_far < self.max_size:
                # 如果当前没满，直接加入最后一个位置
                reservoir_idx = self.n_added_so_far
                self.replace_data(reservoir_idx, stream_data, stream_label)
            # 如果还有位置并且random的数值在容量中，判断后进行替换
            elif reservoir_idx < self.max_size:
                # 计算待替换样本类别中最大的相似度
                buffer_index = reservoir_idx
                # 这里的索引可能不对，需要检查！！！？？？
                buffer_label = self.buffer_labels[buffer_index]
                buffer_indexes = self.get_indexes_of_class(buffer_label)

                # 计算buffer样本的相似度顺序
                model.eval()
                with torch.no_grad():
                    _, buffer_outputs = model(transform(self.buffer_imgs[buffer_indexes.squeeze()].to(self.device)), is_outputs=True)

                # 计算缓冲区中所有样本与均值的累加余弦相似度
                summed_cosine_sim_buffer = torch.zeros(len(buffer_indexes)).to(self.device)
                for param in range(3):
                    # print("param:", param)
                    mean_param = means[buffer_label.item()][param].to(self.device)
                    if param == 0:
                        o_param = 'dts'
                        continue
                    elif param == 1:
                        o_param = 'Bs'
                        # continue
                    else:
                        o_param = 'Cs'
                        continue
                    cosine_sim = F.cosine_similarity(buffer_outputs[o_param], mean_param.unsqueeze(0), dim=1)
                    summed_cosine_sim_buffer += cosine_sim

                # 找到缓冲区中累加余弦相似度最大的样本
                min_sim, min_idx = summed_cosine_sim_buffer.min(dim=0)

                if buffer_label == stream_label:
                    model.eval()
                    with torch.no_grad():
                        _, new_output = model(transform(stream_data.to(self.device).unsqueeze(0)), is_outputs=True)
                    summed_cosine_sim_new = torch.zeros(1).to(self.device)
                    for param in range(3):
                        mean_param = means[stream_label.item()][param].to(self.device)
                        if param == 0:
                            o_param = 'dts'
                            continue
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                        else:
                            o_param = 'Cs'
                            continue
                        cosine_sim = F.cosine_similarity(new_output[o_param].squeeze(), mean_param, dim=0)
                        summed_cosine_sim_new += cosine_sim
                    # 如果当前样本的相似度更小，就不进行替换了
                    if summed_cosine_sim_new.item() < min_sim.item():
                        self.n_seen_so_far += 1
                        continue
            
                reservoir_idx = buffer_indexes[min_idx].item()
                self.replace_data(reservoir_idx, stream_data, stream_label)

            # Increment the seen data counter
            self.n_seen_so_far += 1

    def max1_mamba_update(self, means, model, imgs, labels, transform, **kwargs):
        # outputs
        for stream_data, stream_label in zip(imgs, labels):
            stream_data = stream_data.cpu()
            stream_label = stream_label.cpu()
            if self.n_added_so_far < self.max_size:
                self.stack_data(stream_data, stream_label)
            else:
                max_img_per_class = self.my_get_max_img_per_class()
                class_indexes = self.get_indexes_of_class(stream_label)
                
                # Do nothing if class has reached maximum number of images
                # 如果当前类别的样本数小于允许的每个类别最大样本数
                if len(class_indexes) < max_img_per_class:
                    # Drop img of major class if not
                    major_class = self.get_major_class()
                    class_indexes = self.get_indexes_of_class(major_class)

                    model.eval()
                    with torch.no_grad():
                        _, outputs = model(transform(self.buffer_imgs[class_indexes.squeeze()].to(self.device)), is_outputs=True)

                    summed_cosine_sim = torch.zeros(len(class_indexes)).to(self.device)

                    for param in range(3):
                        # print验证一下！
                        mean_param = means[major_class][param].to(self.device)
                        if param == 0:
                            o_param = 'dts'
                            # continue
                            # print("use dts!!!")
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                            # print("use Bs!!!")
                        else:
                            o_param = 'Cs'
                            # continue
                            # print("use Cs!!!")
                        # shape [N, 1024]  [1, 1024]
                        cosine_sim = F.cosine_similarity(outputs[o_param], mean_param.unsqueeze(0), dim=1)
                        summed_cosine_sim += cosine_sim

                    # Find the index with the maximum summed cosine similarity
                    idx = class_indexes.squeeze()[summed_cosine_sim.argmax()]

                    # Replace the selected data in the buffer
                    self.replace_data(idx, stream_data, stream_label)
                # 如果当前类别的样本数大于等于允许的每个类别最大样本数
                else:
                    # 当前类别已达到最大样本数，需判断是否替换缓冲区中的样本
                    # 获取当前类别在缓冲区中的所有索引
                    class_indexes = self.get_indexes_of_class(stream_label)

                    model.eval()
                    with torch.no_grad():
                        # 获取新样本的模型输出
                        _, new_output = model(transform(stream_data.to(self.device).unsqueeze(0)), is_outputs=True)
                        # 获取缓冲区中当前类别所有样本的模型输出
                        # print("class_indexes.squeeze(): ", class_indexes.squeeze().shape)
                        # print("class_indexes: ", class_indexes.shape)
                        _, buffer_outputs = model(transform(self.buffer_imgs[class_indexes.squeeze()].to(self.device)), is_outputs=True)

                    # 计算新样本与均值的累加余弦相似度
                    new_summed_cosine_sim = torch.zeros(1).to(self.device)
                    for param in range(3):
                        # print("means[stream_label]: ", means[stream_label.item()])
                        mean_param = means[stream_label.item()][param].to(self.device)
                        if param == 0:
                            o_param = 'dts'
                            # continue
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                        else:
                            o_param = 'Cs'
                            # continue
                        cosine_sim = F.cosine_similarity(new_output[o_param].squeeze(), mean_param, dim=0)
                        new_summed_cosine_sim += cosine_sim

                    # 计算缓冲区中所有样本与均值的累加余弦相似度
                    summed_cosine_sim_buffer = torch.zeros(len(class_indexes)).to(self.device)
                    for param in range(3):
                        mean_param = means[stream_label.item()][param].to(self.device)
                        if param == 0:
                            o_param = 'dts'
                            # continue
                        elif param == 1:
                            o_param = 'Bs'
                            # continue
                        else:
                            o_param = 'Cs'
                            # continue
                        cosine_sim = F.cosine_similarity(buffer_outputs[o_param], mean_param.unsqueeze(0), dim=1)
                        summed_cosine_sim_buffer += cosine_sim

                    # 找到缓冲区中累加余弦相似度最大的样本
                    max_sim, max_idx = summed_cosine_sim_buffer.max(dim=0)

                    # 如果新样本的相似度小于缓冲区中最大的相似度，则进行替换
                    if new_summed_cosine_sim.item() <= max_sim.item():
                        # 获取需要替换的样本在缓冲区中的实际索引
                        idx_to_replace = class_indexes[max_idx].item()
                        self.replace_data(idx_to_replace, stream_data, stream_label)
                    
            # Increment the seen data counter
            self.n_seen_so_far += 1

    def min1_mamba_update(self, means, model, imgs, labels, transform, **kwargs):
        for stream_data, stream_label in zip(imgs, labels):
            # print("self.n_added_so_far=", self.n_added_so_far)
            stream_data = stream_data.cpu()
            stream_label = stream_label.cpu()
            if self.n_added_so_far < self.max_size:
                self.stack_data(stream_data, stream_label)
            else:
                max_img_per_class = self.my_get_max_img_per_class()
                class_indexes = self.get_indexes_of_class(stream_label)
                
                # Do nothing if class has reached maximum number of images
                # 如果当前类别的样本数小于等于允许的每个类别最大样本数
                if len(class_indexes) < max_img_per_class:
                    # Drop img of major class if not
                    major_class = self.get_major_class()
                    class_indexes = self.get_indexes_of_class(major_class)

                    # compute cosine similarity to mean for multiple parameters
                    model.eval()
                    with torch.no_grad():
                        # Get the model outputs dictionary
                        _, outputs = model(transform(self.buffer_imgs[class_indexes.squeeze()].to(self.device)), is_outputs=True)
                        # outputs is a dictionary containing 'dts', 'Bs', 'Cs'

                    # Initialize a tensor to store the summed cosine similarities
                    summed_cosine_sim = torch.zeros(len(class_indexes)).to(self.device)

                    # Iterate over the three parameters: 'dts', 'Bs', 'Cs'
                    for param in range(3):
                        # Retrieve the mean vector for the current parameter and class
                        mean_param = means[major_class][param].to(self.device)
                        # Compute cosine similarity between outputs and mean
                        if param == 0:
                            o_param = 'dts'
                        elif param == 1:
                            o_param = 'Bs'
                        else:
                            o_param = 'Cs'
                        cosine_sim = F.cosine_similarity(outputs[o_param], mean_param.unsqueeze(0), dim=1)
                        # Add the cosine similarity to the summed result
                        summed_cosine_sim += cosine_sim

                    # Find the index with the minimum summed cosine similarity
                    idx = class_indexes.squeeze()[summed_cosine_sim.argmin()]

                    # Replace the selected data in the buffer
                    self.replace_data(idx, stream_data, stream_label)
                # 如果当前类别的样本数大于等于允许的每个类别最大样本数
                else:
                    # 当前类别已达到最大样本数，需判断是否替换缓冲区中的样本
                    # 获取当前类别在缓冲区中的所有索引
                    class_indexes = self.get_indexes_of_class(stream_label)

                    model.eval()
                    with torch.no_grad():
                        # 获取新样本的模型输出
                        _, new_output = model(transform(stream_data.to(self.device).unsqueeze(0)), is_outputs=True)
                        # 获取缓冲区中当前类别所有样本的模型输出
                        _, buffer_outputs = model(transform(self.buffer_imgs[class_indexes.squeeze()].to(self.device)), is_outputs=True)

                    # 计算新样本与均值的累加余弦相似度
                    new_summed_cosine_sim = torch.zeros(1).to(self.device)
                    for param in range(3):
                        # print("means[stream_label]: ", means[stream_label.item()])
                        mean_param = means[stream_label.item()][param].to(self.device)
                        if param == 0:
                            o_param = 'dts'
                        elif param == 1:
                            o_param = 'Bs'
                        else:
                            o_param = 'Cs'
                        cosine_sim = F.cosine_similarity(new_output[o_param], mean_param.unsqueeze(0), dim=1)
                        new_summed_cosine_sim += cosine_sim

                    # 计算缓冲区中所有样本与均值的累加余弦相似度
                    summed_cosine_sim_buffer = torch.zeros(len(class_indexes)).to(self.device)
                    for param in range(3):
                        mean_param = means[stream_label.item()][param].to(self.device)
                        if param == 0:
                            o_param = 'dts'
                        elif param == 1:
                            o_param = 'Bs'
                        else:
                            o_param = 'Cs'
                        cosine_sim = F.cosine_similarity(buffer_outputs[o_param], mean_param.unsqueeze(0), dim=1)
                        summed_cosine_sim_buffer += cosine_sim

                    # 找到缓冲区中累加余弦相似度最小的样本
                    min_sim, min_idx = summed_cosine_sim_buffer.min(dim=0)

                    # 如果新样本的相似度大于缓冲区中最小的相似度，则进行替换
                    if new_summed_cosine_sim.item() > min_sim.item():
                        # 获取需要替换的样本在缓冲区中的实际索引
                        idx_to_replace = class_indexes[min_idx].item()
                        # 替换缓冲区中的样本
                        self.replace_data(idx_to_replace, stream_data, stream_label)
                    # 否则，不进行任何操作
                    
            # Increment the seen data counter
            self.n_allseen_so_far += 1
