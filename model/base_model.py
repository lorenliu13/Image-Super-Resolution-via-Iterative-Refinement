import os
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        """
        Function that moves the input data to specified device (GPU or CPU)
        x: can be dictionary, list, or single tensor
        """
        if isinstance(x, dict):
            # if x is a dictionary
            for key, item in x.items():
                # iterate through the dictionary's key-value pairs
                if item is not None:
                    # if the item is not none, move it to the target device
                    x[key] = item.to(self.device) # move it to the target device
        elif isinstance(x, list):
            # if it is a list
            for item in x:
                # iterate through the list
                if item is not None:
                    item = item.to(self.device) # move it to the target device
        else:
            x = x.to(self.device) # if it is a tensor, move to the target device
        return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
