import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from aspp import DeepLabHead
from create_dataset import NYUv2
from torch.utils.data import Subset
from LibMTL import Trainer
from LibMTL.model import resnet_dilated
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
import wandb
from TwoSampler import *
# from data_utils import *

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--train_bs', default=8, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/home/saiteja/LibMTL/nyuv2', type=str, help='dataset path')
    return parser.parse_args()
    
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    nyuv2_test_set = NYUv2(root=params.dataset_path, mode='test', augmentation=False)
    labeled_dataset = NYUv2(root=params.dataset_path, mode='labeled', augmentation=False)
    unlabeled_dataset = NYUv2(root=params.dataset_path, mode='unlabeled', augmentation=False)

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=params.train_bs, shuffle=True)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=params.train_bs, shuffle=False)
    
    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    
    # define tasks
    task_dict = {'depth': {'metrics':['abs_err', 'rel_err'], 
                           'metrics_fn': DepthMetric(),
                           'loss_fn': DepthLoss(),
                           'weight': [0, 0]},
                 'normal': {'metrics':['mean', 'median', '<11.25', '<22.5', '<30'], 
                            'metrics_fn': NormalMetric(),
                            'loss_fn': NormalLoss(),
                            'weight': [0, 0, 1, 1, 1]}}
    
    # define encoder and decoders
    def encoder_class(): 
        return resnet_dilated('resnet50')
    num_out_channels = {'depth': 1, 'normal': 3}
    decoders = nn.ModuleDict({task: DeepLabHead(2048, 
                                                num_out_channels[task]) for task in list(task_dict.keys())})
    
    class NYUtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
            super(NYUtrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting_method.__dict__[weighting], 
                                            architecture=architecture_method.__dict__[architecture], 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)

        def process_preds(self, preds):
            img_size = (288, 384)
            for task in self.task_name:
                preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
            return preds
    
    NYUmodel = NYUtrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=encoder_class, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    if params.mode == 'train':
        # NYUmodel.train(nyuv2_train_loader, nyuv2_test_loader, params.epochs)
        NYUmodel.train_sl(labeled_loader, nyuv2_test_loader, params.epochs,unlabeled_dataloader=unlabeled_loader,bs=8, labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset)
    elif params.mode == 'test':
        NYUmodel.test(nyuv2_test_loader)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    wandb.init(project='omni', config=params)
    config = wandb.config
    config.aug = params.aug
    config.train_mode = params.train_mode
    config.train_bs = params.train_bs
    config.test_bs = params.test_bs
    config.epochs = params.epochs
    main(params)
