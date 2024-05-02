import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 2048, 36, 48)

class AbsArchitecture(nn.Module):
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(AbsArchitecture, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.encoder_class = encoder_class
        self.decoders = decoders
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.device = device
        self.kwargs = kwargs
        
        # Add a fully connected layer and reshape operation
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 2048*36*48)
        self.reshape = Reshape()
        
        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}
    
    def forward(self, inputs, task_name=None):
        r"""

        Args: 
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.
        
        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        out = {}
        s_rep = self.encoder(inputs)
        
        # Apply the fully connected layer and reshape operation if necessary
        s_rep = self.fc2(self.fc1(s_rep))
        s_rep = self.reshape(s_rep)
        
        # print(f'the shape of the input after encoder is{s_rep.shape}')
        same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, task, same_rep)
            out[task] = self.decoders[task](ss_rep)
        return out

    
    def get_share_params(self):
        r"""Return the shared parameters of the model.
        """
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.encoder.zero_grad(set_to_none=False)
        
    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep
