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
    parser.add_argument('--dataset_path', default='/home/g054545/LibMTL', type=str, help='dataset path')
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
    task_dict = {'segmentation': {'metrics':['mIoU', 'pixAcc'], 
                              'metrics_fn': SegMetric(),
                              'loss_fn': SegLoss(),
                              'weight': [1, 1]}}
    
    import vision_transformer as vvv
    from urllib.parse import urlparse
    def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
        if urlparse(pretrained_weights).scheme:  # If it looks like an URL
            state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
        else:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        return model
        
    def encoder_class():
        model = load_pretrained_weights(vvv.vit_large(patch_size=14),"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_ade20k_linear_head.pth","teacher")
        # model.cuda()
        print(model)
        return model  
        # return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') 
        # return resnet_dilated('resnet50')
    num_out_channels = {'segmentation': 14}
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
            img_size = (280, 280)
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
