from random import random
import torch
import logging as lg
import os
import pickle
import time
import os
import datetime as dt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random as r
import torchvision
import wandb
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from torchvision import transforms
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomInvert
from copy import deepcopy
from torchvision.transforms import RandAugment
from umap import UMAP
from torch.distributions import Categorical

from src.utils.utils import save_model
from src.models import resnet
from src.utils.metrics import forgetting_line 
from src.utils.utils import get_device

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

device = get_device()

class BaseLearner(torch.nn.Module):
    def __init__(self, args):
        """Learner abstract class
        Args:
            args: argparser arguments
        """
        super().__init__()
        self.params = args
        self.multiplier = self.params.multiplier
        self.device = get_device()
        self.init_tag()
        # self.model = self.load_model()
        # self.optim = self.load_optim()
        self.buffer = None
        self.start = time.time()
        self.criterion = self.load_criterion()
        if self.params.tensorboard:
            self.writer = self.load_writer()
        self.classifiers_list = ['ncm']  # Classifiers used for evaluating representation
        self.loss = 0
        self.stream_idx = 0
        self.results = []
        self.results_etf = []
        self.results_combine = []
        self.results_clustering = []
        self.results_forgetting = []
        self.results_clustering_forgetting = []
        self.results_forgetting_etf = []
        self.results_forgetting_combine = []
        
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.params.dataset == 'tiny' else nn.Identity()
        normalize = nn.Identity()
        if self.params.tf_type == 'partial':
            print("trans1!!!")
            self.transform_train = nn.Sequential(
                torchvision.transforms.RandomCrop(self.params.img_size, padding=4),
                RandomHorizontalFlip(),
                normalize
            ).to(device)
        elif self.params.tf_type == 'moderate':
            print("trans2!!!")
            self.transform_train = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(0.6, 1.)),
                RandomHorizontalFlip(),
                RandomGrayscale(p=0.2),
                normalize
            ).to(device)
        else:
            print("trans3!!!")
            self.transform_train = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(self.params.min_crop, 1.)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
                normalize
            ).to(device)

        self.transform_test = nn.Sequential(
            normalize
        ).to(device)

    def init_tag(self):
        """Initialise tag for experiment
        """
        if self.params.training_type == 'inc':
            self.params.tag = f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}{self.params.tag}"
        elif self.params.training_type == "blurry":
            self.params.tag =\
                 f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}blurry{self.params.blurry_scale}{self.params.tag}"
        else:
            self.params.tag = f"{self.params.learner},{self.params.dataset},{self.params.epochs}b{self.params.batch_size},uni{self.params.tag}"
        print(f"Using the following tag for this experiment : {self.params.tag}")
    
    def load_writer(self):
        """Initialize tensorboard summary writer
        """
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(self.params.tb_root, self.params.tag))
        return writer

    def save(self, path):
        lg.debug("Saving checkpoint...")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))
        
        with open(os.path.join(path, f"memory.pkl"), 'wb') as memory_file:
            pickle.dump(self.buffer, memory_file)

    def resume(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        with open(os.path.join(path, f"memory.pkl"), 'rb') as f:
            self.buffer = pickle.load(f)
        f.close()
        torch.cuda.empty_cache()


    def load_model(self):
        """Load model
        Returns:
            untrained torch backbone model
        """
        return NotImplementedError


    def load_optim(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        if self.params.optim == 'Adam':
            if self.params.learner == 'OnPro':
                print("Adam with OnPro!!!!!!!!!!!")
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate, betas=(0.9, 0.99), weight_decay=self.params.weight_decay)
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
                )
        else: 
            raise Warning('Invalid optimizer selected.')
        return optimizer

    def load_scheduler(self):
        raise NotImplementedError

    def load_criterion(self):
        raise NotImplementedError
    
    def train(self, dataloader, task, **kwargs):
        raise NotImplementedError

    def plot_embedding_2d(self, X: torch.Tensor, y, save_path, title=None):
        """
        Plot a 2D embedding X with the class label y colored by the domain and save it as a PDF.

        Parameters:
        X : numpy.ndarray or torch.Tensor
            The 2D coordinates to plot. Can be a tensor with more dimensions that needs flattening.
        y : array-like
            The class labels corresponding to each point in X.
        title : str, optional
            The title of the plot.
        save_path : str, optional
            The file path to save the plot as a PDF.
        """
        # If X is a tensor with more than 2 dimensions, flatten it
        # print("X", X)
        # print("X", X.shape)
        # If X is a tensor with more than 2 dimensions, flatten it
        if isinstance(X, torch.Tensor):
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            # Convert tensor to numpy
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

        elif isinstance(X, np.ndarray):
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
        else:
            raise TypeError("Input X must be a numpy.ndarray or torch.Tensor")

        # Explicit normalization to range [0, 1]
        # x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
        # X = (X - x_min) / (x_max - x_min + 1e-8)  # Add epsilon to avoid division by zero

        # Reduce dimensionality to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=0, learning_rate=50, perplexity=5, init='pca', n_iter=50000)
        X = tsne.fit_transform(X)

        # Normalize t-SNE output to range [0, 1]
        x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
        X = (X - x_min) / (x_max - x_min)  # Add epsilon to avoid division by zero

        # Create the figure
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)

        # Plot each point as a solid circle with distinct colors
        for i in range(X.shape[0]):
            plt.scatter(X[i, 0], X[i, 1], color=plt.cm.tab20(y[i] % 20), s=50, alpha=0.7)

        # Remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set title if provided
        if title is not None:
            plt.title(title)

        # Save the plot as a PDF
        with PdfPages(save_path) as pdf:
            pdf.savefig()  # saves the current figure into a pdf page
            # Make confusion matrix
            if not self.params.no_wandb:
                wandb.log({"embedding_plot": wandb.Image(plt, caption=title or "Embedding Visualization")})

            plt.close()  # close the figure to free memory

    # evaluate the model on the test set
    
    def evaluate(self, dataloaders, task_id, eval_ema=False, **kwargs):
        with torch.no_grad():
            self.model.eval()
            accs = []
            preds = []
            all_targets = []
            tag = '' 
            for j in range(task_id + 1):
                test_preds, test_targets = self.encode_logits(dataloaders[f"test{j}"])
                acc = accuracy_score(test_targets, test_preds)

                accs.append(acc)
                # Wandb logs
                if not self.params.no_wandb:
                    preds = np.concatenate([preds, test_preds])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"acc_{j}": acc,
                        "task_id": task_id
                    })
            
            # Make confusion matrix
            if not self.params.no_wandb:
                # re-index to have classes in task order
                all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                preds= [self.params.labels_order.index(int(i)) for i in preds]
                cm= np.log(1 + confusion_matrix(all_targets, preds))
                fig = plt.matshow(cm)
                wandb.log({
                        tag + f"cm": fig,
                        "task_id": task_id
                    })
                
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)

            self.results.append(accs)
            
            line = forgetting_line(pd.DataFrame(self.results), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting.append(line)

            self.print_results(task_id)

            return np.nanmean(self.results[-1]), np.nanmean(self.results_forgetting[-1]) 

    def evaluate_offline(self, dataloaders, epoch=0, eval_ema=False):
        with torch.no_grad():
            self.model.eval()
            test_preds, test_targets = self.encode_logits(dataloaders['test'])
            acc= accuracy_score(test_targets, test_preds)
            self.results.append(acc)

        print(f"ACCURACY {self.results[-1]}")
        return self.results[-1]
    
    def evaluate_clustering(self, dataloaders, task_id, eval_ema=False, **kwargs):
        try:
            results = self._evaluate_clustering(dataloaders, task_id, eval_ema=eval_ema, **kwargs)
        except:
            results = 0, 0
        return results

    # use classifier to evaluate the model
    def _evaluate_clustering(self, dataloaders, task_id, eval_ema=False, **kwargs):
        with torch.no_grad():
            self.model.eval()

            # Train classifier on labeled data
            step_size = int(self.params.n_classes/self.params.n_tasks)
            mem_representations, mem_labels = self.get_mem_rep_labels(use_proj=self.params.eval_proj)

            # UMAP visualization
            # reduction = self.umap_reduction(mem_representations.cpu().numpy())
            # plt.figure()
            # figure = plt.scatter(reduction[:, 0], reduction[:, 1], c=mem_labels, cmap='Spectral', s=1)
            # if not self.params.no_wandb:
            #     wandb.log({
            #         "umap": wandb.Image(figure),
            #         "task_id": task_id
            #     })

            
            classifiers= self.init_classifiers()
            classifiers= self.fit_classifiers(classifiers=classifiers, representations=mem_representations, labels=mem_labels)
            
            accs = []
            representations= {}
            targets= {}
            preds= []
            all_targets = []
            tag = ''

            for j in range(task_id + 1):
                test_representation, test_targets = self.encode_fea(dataloaders[f"test{j}"])
                representations[f"test{j}"] = test_representation
                targets[f"test{j}"] = test_targets

                test_preds= classifiers[0].predict(representations[f'test{j}'])

                acc= accuracy_score(targets[f"test{j}"], test_preds) 

                accs.append(acc)
                # Wandb logs
                if not self.params.no_wandb:
                    preds= np.concatenate([preds, test_preds])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"ncm_acc_{j}": acc,
                        "task_id": task_id
                    })
            
            # Make confusion matrix
            if not self.params.no_wandb:
                # re-index to have classes in task order
                all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                preds= [self.params.labels_order.index(int(i)) for i in preds]
                cm= np.log(1 + confusion_matrix(all_targets, preds))
                fig = plt.matshow(cm)
                wandb.log({
                        tag + f"ncm_cm": fig,
                        "task_id": task_id
                    })

                
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)

            self.results_clustering.append(accs)
            
            line = forgetting_line(pd.DataFrame(self.results_clustering), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_clustering_forgetting.append(line)

            self.print_results(task_id)

            return np.nanmean(self.results_clustering[-1]), np.nanmean(self.results_clustering_forgetting[-1])

    # 只是生成features,也用了线性层分类，没有用argmax
    def encode_fea(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)

                feat = self.model(self.transform_test(inputs))

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat= feat.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat= np.vstack([all_feat, feat.cpu().numpy()])

        return all_feat, all_labels

    # 生成features之后顺便用线性层进行分类，返回结果就是分类结果
    def encode_logits(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)

                feat = self.model.logits(self.transform_test(inputs))

                # choose the max prediction
                preds= feat.argmax(dim=1)

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat= preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat= np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels

    # return different classifiers
    # ??? without backpropagation
    def init_classifiers(self):
        """Initiliaze every classifier for representation transfer learning
        Returns:
            List of initialized classifiers
        """
        # logreg = LogisticRegression(random_state=self.params.seed)
        # knn = KNeighborsClassifier(3)
        # linear = MLPClassifier(hidden_layer_sizes=(200), activation='identity', max_iter=500, random_state=self.params.seed)
        svm = SVC() # better performence
        ncm = NearestCentroid()
        # return [logreg, knn, linear, svm, ncm]
        return [ncm, svm]
    
    @ignore_warnings(category=ConvergenceWarning)
    def fit_classifiers(self, classifiers, representations, labels):
        """Fit every classifiers on representation - labels pairs
        Args:
            classifiers : List of sklearn classifiers
            representations (torch.Tensor): data representations 
            labels (torch.Tensor): data labels
        Returns:
            List of trained classifiers
        """
        for clf in classifiers:
            clf.fit(representations.cpu(), labels)
        return classifiers

    def after_eval(self, **kwargs):
        pass

    def before_eval(self, **kwargs):
        pass

    def train_inc(self, **kwargs):
        raise NotImplementedError

    def train_blurry(self, **kwargs):
        raise NotImplementedError

    def get_mem_rep_labels(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        if eval: 
            self.model.eval()
        mem_imgs, mem_labels = self.buffer.get_all()
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            mem_representations_b = self.model(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations= torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels

    def save_results(self):
        pass
    
    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")


    def umap_reduction(self, representation):
        umap_reducer = UMAP()
        umap_result = umap_reducer.fit_transform(representation)
        return umap_result

    def backward_transfer(self):
        n_tasks = len(self.results)
        bt = 0
        for i in range(1, n_tasks):
            for j in range(i):
                bt += self.results[i][j] - self.results[j][j]
        
        return bt / (n_tasks * (n_tasks - 1) / 2)
                
    def learning_accuracy(self):
        n_tasks = len(self.results)
        la = 0
        for i in range(n_tasks):
                la += self.results[i][i]
        return la / n_tasks
    
    def reletive_forgetting(self):
        n_tasks = len(self.results)
        rf = 0
        max = np.nanmax(np.array(self.results), axis=0)
        for i in range(n_tasks-1):
            if max[i] != 0:
                rf += self.results_forgetting[-1][i] / max[i]
            else:
                rf += 1
        
        return rf / n_tasks
    
    def get_entropy(self, dataloaders, task_id):
        trainloader = dataloaders[f"train{task_id}"]
        testloader = dataloaders[f"test{task_id}"]

        train_ce = 0
        train_en = 0
        test_ce = 0
        test_en = 0
        samples = 0

        self.model.eval()

        for i, batch in enumerate(trainloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device).long()
            samples += inputs.shape[0]
            outputs = self.model.logits(self.transform_test(inputs))
            prob = torch.softmax(outputs, dim=1)
            train_ce += torch.nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
            train_en += Categorical(probs=prob).entropy().sum().item()

        train_ce /= samples
        train_en /= samples

        samples = 0

        for i, batch in enumerate(testloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device).long()
            samples += inputs.shape[0]
            outputs = self.model.logits(self.transform_test(inputs))
            prob = torch.softmax(outputs, dim=1)
            test_ce += torch.nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
            test_en += Categorical(probs=prob).entropy().sum().item()

        test_ce /= samples
        test_en /= samples

        self.model.train()
        return train_ce, train_en, test_ce, test_en

    def etf_predict(self, feat, target):
        # 将点积值转化为预测结果
        # 假设我们有多个类别，我们希望获取每个类别的点积值：
        # 注意：假设 target 是 ETF 分类器的中心参数矩阵
        all_dots = torch.matmul(feat, target)  # [batch_size, num_classes]
        # 获取预测结果
        y_pred_etf = F.softmax(all_dots, dim=1)  # [batch_size, num_classes]
        return y_pred_etf

    def calculate_cons_loss(self, outputs, y):
        sim = 0.0
        sim += self.calculate_similarity_matrix(outputs['dts'], y)   
        return sim

    def calculate_similarity_matrix(self, vectors, labels):
        B, D = vectors.shape
        sim = 0.0
        
        # 计算每一对向量的余弦相似度
        for i in range(B):
            for j in range(i+1, B):                
                # 计算余弦相似度
                cos_sim = F.cosine_similarity(vectors[i], vectors[j], dim=0) # 不能直接处理3D的数据，只能处理某一维，目前将后面三维直接展平
                
                # # 相似度本身有正负
                if labels[i] == labels[j]:
                    sim -= cos_sim
                #     print("same:", cos_sim)
                # # 或许不需要负数的部分
                # if labels[i] != labels[j]:
                #     sim += cos_sim
                #     print("diff:", cos_sim)

                # 相似度本身有正负
                # if labels[i] == labels[j]:
                #     sim += 1.0-cos_sim
                    # print("same:", cos_sim)
                # 或许不需要负数的部分
                # if labels[i] != labels[j]:
                #     sim += cos_sim+1.0
                    # print("diff:", cos_sim)

        
        return sim

    def evaluate_etf(self, dataloaders, task_id, eval_ema=False, **kwargs):
        with torch.no_grad():
            self.model.eval()  # Set the model to evaluation mode
            accs = []
            preds = []
            all_targets = []
            tag = '' 
            for j in range(task_id + 1):
                # 获取模型的预测结果和真实标签
                test_preds, test_targets = self.encode_logits_etf(dataloaders[f"test{j}"])
                acc = accuracy_score(test_targets, test_preds)
                accs.append(acc)
                # Wandb logs
                if not self.params.no_wandb:
                    preds = np.concatenate([preds, test_preds])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"ETFacc_{j}": acc,
                        "task_id": task_id
                    })           
            # todo
            # 绘制混淆矩阵
            # 填充缺失的任务准确率
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)

            self.results_etf.append(accs)
            
            # 计算遗忘率
            line = forgetting_line(pd.DataFrame(self.results_etf), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting_etf.append(line)
            self.print_results(task_id)
            return np.nanmean(self.results_etf[-1]), np.nanmean(self.results_forgetting_etf[-1])

    def encode_logits_etf(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)
                # todo:加入etf_transform
                current_mask_batch = torch.zeros_like(labels, dtype=torch.float).to(device)

                feat = self.pre_logits(self.etf_transform(self.model(self.transform_test(inputs))))
                scores = torch.matmul(feat, self.etf_classifier) 
                _, preds = torch.max(scores, dim=1)
                # preds,_  = torch.max(scores, dim=1)

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat= preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat= np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels

    def evaluate_combine(self, dataloaders, task_id, eval_ema=False, **kwargs):
        with torch.no_grad():
            self.model.eval()  # Set the model to evaluation mode
            accs = []
            preds = []
            all_targets = []
            tag = '' 
            for j in range(task_id + 1):
                # 获取模型的预测结果和真实标签
                test_preds, test_targets = self.encode_logits_combine(dataloaders[f"test{j}"])
                acc = accuracy_score(test_targets, test_preds)
                accs.append(acc)
                # Wandb logs
                if not self.params.no_wandb:
                    preds = np.concatenate([preds, test_preds])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"ETFacc_{j}": acc,
                        "task_id": task_id
                    })           
            # todo
            # 绘制混淆矩阵
            # 填充缺失的任务准确率
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)

            self.results_combine.append(accs)
            
            # 计算遗忘率
            line = forgetting_line(pd.DataFrame(self.results_combine), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting_combine.append(line)
            self.print_results(task_id)
            return np.nanmean(self.results_combine[-1]), np.nanmean(self.results_forgetting_combine[-1])

    def encode_logits_combine(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)
                # todo:加入etf_transform
                current_mask_batch = torch.zeros_like(labels, dtype=torch.float).to(device)

                feat_etf = self.pre_logits(self.etf_transform(self.model(self.transform_test(inputs))))
                scores = torch.matmul(feat_etf, self.etf_classifier) 
                scores = F.softmax(scores, dim=1)
                # print("scores:", scores)
                zhi_etf, preds_etf = torch.max(scores, dim=1)
                # preds,_  = torch.max(scores, dim=1)

                feat_linear = self.model.logits(self.transform_test(inputs))
                feat_linear = F.softmax(feat_linear, dim=1)
                # print("feat_linear:", feat_linear)
                zhi_linear, preds_linear = torch.max(feat_linear, dim=1)

                final_preds = torch.where(zhi_etf > zhi_linear, preds_etf, preds_linear)
                final_confidence = torch.where(zhi_etf > zhi_linear, zhi_etf, zhi_linear)
                # print("final_confidence:", final_confidence)

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat= final_preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat= np.hstack([all_feat, final_preds.cpu().numpy()])
        return all_feat, all_labels



