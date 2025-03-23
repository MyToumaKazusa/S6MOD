import os
import torch
import logging as lg
import time
import torch.nn.functional as F
import numpy as np

from torch import nn
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

class Logger(object):
    def __init__(self, args, base_dir='./outputs'):

        self.args = args
        if args.wandb_log == 'online':
            import wandb
            wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                name    = args.run_name,
                config  = args
            )
            self.wandb = wandb
        else:
            self.wandb = None

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.base_dir = base_dir

        self.folder_path = os.path.join(base_dir, args.run_name+f'_{np.random.randint(1000)}')
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)

        # dump args
        with open(os.path.join(self.folder_path, "params.json"), "wt") as f:
            json.dump(vars(args), f)

        self.to_pickle  = []
        self.picklename = os.path.join(self.folder_path,  "db.pickle")

    def log_scalars(self, values, step, verbose=False):
        for k,v in values.items():
            self.to_pickle += [(k, v, step)]
        if verbose:
            print(values)
        if self.wandb is None:
            return
        self.wandb.log(values, step)

    def log_losses(self, loss_log_holder):
        if self.wandb is None:
            return
        for loss_log in loss_log_holder:
            step = loss_log.pop('step')
            self.log_scalars(loss_log, step, verbose=False)

    def log_accs(self, acc_log):
        step = acc_log.pop('step')
        for k, v in acc_log.items():
            self.to_pickle += [(k, v, step)]
        if self.wandb is None:
            return
        for k, v in acc_log.items():
            for i in range(len(v)):
                self.wandb.log({f"{k}/{i}": v[i]}, step)

    def log_accs_table(self, name, accs_list, step, verbose=False):

        num_tasks, _ = accs_list.shape
        col_name = [f"task{i}" for i in range(num_tasks)]
        accs_table = pd.DataFrame(accs_list, columns=col_name, index=np.array(col_name))
        if verbose:
            print(accs_table)

        accs_table.to_csv(os.path.join(self.folder_path, f"{name}.csv"))

        if self.wandb is None:
            return
        self.wandb.log({name: self.wandb.Table(dataframe=accs_table)}, step)

    def log_img(self, values, step):
        if self.wandb is None:
            return
        for k, fig in values.items():
            figure_path = os.path.join(self.path, f"{k.replace('/', '_')}_step{step}.png")
            fig.savefig(figure_path)
            fig.clear()
            self.wandb.log({k:self.wandb.Image(figure_path)}, step)

    def dump(self):
        f = open(self.picklename, "ab")
        pickle.dump(self.to_pickle, f)
        f.close()
        self.to_pickle = []

    def close(self):
        self.dump()
        if self.wandb is None:
            return
        self.wandb.finish()

def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), t_coef * sem(acc_per_run))


    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1,2)) -
                  np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                         (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.sum(np.triu(end_task_acc_arr, 1), axis=(1,2)) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.mean(fwt_per_run), t_coef * sem(fwt_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt

def get_device():
    if torch.cuda.is_available():
        dev = "cuda"
    # elif torch.backends.mps.is_available():
    #     dev = "mps"
    else:
        dev = "cpu"
    return torch.device(dev)

device = get_device()

def save_model(model_weights, model_name, dir="./checkpoints/"):
    """Save PyTorch model weights
    Args:
        model_weights (Dict): model stat_dict
        model_name (str): name_of_the_model.pth
    """
    lg.debug("Saving checkpoint...")
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    torch.save(model_weights, os.path.join(dir, model_name))

def get_pairs(n_data:int, y=None, mask=None):
    """Get pairs of samples for metric learning.
        mask (torch.Tensor): Mask where mask[i,j] = 1 <=> y_i = y_j. (bsz, bsz). Default: None
    
    """
    if mask is None:
        if y is None:
            mask = torch.eye(n_data).to(device)
        else:
            y = y.view(-1, 1)
            mask = torch.eq(y, y.T).float().to(device)
    mask = mask.repeat(2, 2)
    
    # index of postive and negative pairs
    tri = torch.tril(torch.ones(mask.size()), -1).to(device)
    pos_pairs = torch.vstack(torch.where(torch.logical_and(mask, tri))).T
    neg_pairs = torch.vstack(torch.where(torch.logical_and(torch.logical_not(mask), tri))).T

    return pos_pairs, neg_pairs


# @utils.tensorfy(0, 1, tensor_klass=torch.LongTensor)
def filter_labels(y, labels):
    """Utility used to create a mask to filter values in a tensor.

    Args:
        y (list, torch.Tensor): tensor where each element is a numeric integer
            representing a label.
        labels (list, torch.Tensor): filter used to generate the mask. For each
            value in ``y`` its mask will be "1" if its value is in ``labels``,
            "0" otherwise".

    Shape:
        y: can have any shape. Usually will be :math:`(N, S)` or :math:`(S)`,
            containing `batch X samples` or just a list of `samples`.
        labels: a flatten list, or a 1D LongTensor.

    Returns:
        mask (torch.ByteTensor): a binary mask, with "1" with the respective value from ``y`` is
        in the ``labels`` filter.

    Example::

        >>> a = torch.LongTensor([[1,2,3],[1,1,2],[3,5,1]])
        >>> a
         1  2  3
         1  1  2
         3  5  1
        [torch.LongTensor of size 3x3]
        >>> classification.filter_labels(a, [1, 2, 5])
         1  1  0
         1  1  1
         0  1  1
        [torch.ByteTensor of size 3x3]
        >>> classification.filter_labels(a, torch.LongTensor([1]))
         1  0  0
         1  1  0
         0  0  1
        [torch.ByteTensor of size 3x3]
    """
    mapping = torch.zeros(y.size()).byte()

    for label in labels:
        mapping = mapping | y.eq(label)

    return mapping


def timing(function):
    """Timing decorator for code profiling

    Args:
        function : Function to evaluate (measures time performance)
    """
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time  = time.time()
        duration = (end_time- start_time)*1000.0
        f_name = function.__name__
        lg.info("{} took {:.3f} ms".format(f_name, duration))

        return result
    return wrap


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def AG_SawSeriesPT(y:torch.tensor, sigma2:torch.tensor, d:torch.tensor, N:torch.arange, normalize=True):
    """Implements Saw series in PyTorch

    Args:
        y (torch.tensor): Density value
        sigma2 (torch.tensor):      Standard deviation of the original gaussian
        d (torch.tensor):           Dimensionality of the data
        N (torch.arange):           Number of elements in the (theoriticaly infinite) sum
        normalize (bool, optional): Whether to normalize the density. Defaults to True.

    Returns:
        torch.tensor: Density value
    """
    yk = y/torch.sqrt(sigma2)
    a = 1/torch.sqrt(torch.Tensor([2]).to(device))*yk
    s = 0
    for k in N:
        k = k.to(device)
        # w1 = ((2*a)**k)*torch.special.gammaln((d+k)/2).exp()/(torch.special.gammaln(k+1).exp()*torch.special.gammaln(d/2).exp())
        w = ((2*a)**k)*(torch.special.gammaln((d+k)/2) - torch.special.gammaln(k+1) - torch.special.gammaln(d/2)).exp()
        s = s + w
    p = s*torch.exp(-1/2*(1/sigma2)).to(device)
    if not normalize: 
        S = (2*(torch.pi)**(d/2))/torch.special.gammaln(d/2).exp().to(device)
        p = p/S

    return p.to(device)


def make_orthogonal(U, labels=None):
    """Orthogonalisation of features matrix by Giovanni.

    Args:
        U (torch.Tensor): Feature matrix of shape (batch_size, feature_dim)
        labels (torch.Tensor, optional): Feature labels. Defaults to None.

    Returns:
        torch.Tensor: orthogonalized matrix.
    """
    # Need to add a small value on the diagonal to ensure the matrix is postiive semi-definite
    # This is due to approximation error
    M = (U.mT @ U) + (torch.eye(U.shape[1]) * 1e-3).to(device)
    R = torch.cholesky(M, upper=True)
    Q = U @ torch.inverse(R)
    if labels is None:
        Q = Q / Q.max(dim=0)[0]
    else:
        bins = torch.bincount(labels)
        c = bins[labels].unsqueeze(1)
        Q = Q * torch.sqrt(c)
    return Q