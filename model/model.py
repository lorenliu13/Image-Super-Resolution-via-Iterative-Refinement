import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel): # inherits from "Basemodel"
    """

    """
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt)) # defines the generator netwrok
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train') # sets the noise schedule
        if self.opt['phase'] == 'train':
            # sets the generator network to training mode
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            # initialize an Adam optimizer with learning rate
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network() # load the model from a checkpoint
        self.print_network() # print the structure of the network

    def feed_data(self, data):
        """
        Feed the image data to the model
        """
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad() # set the gradients of the generator's optimizer to zero
        # compute the loss value by passing the input data through the generator network (netG)
        # loss represents the difference between the predicted output and the ground truth
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape # extract the dimensions of input data (batch, channel, height, width)
        # compute the average loss value
        l_pix = l_pix.sum()/int(b*c*h*w)
        # perform backpropagation, calculates the gradients of loss with respect to the generator's parameters
        l_pix.backward()
        # update the generator network's parameters using the calculated gradients
        self.optG.step()

        # set log
        # stores the computed loss value in the log_dict
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        """
        Perform a single inference step with the diffusion model to generate SR images.
        continous: whether to return multiple images showing the sampling processes
        """

        self.netG.eval() # set the diffusion model to evaluation mode
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel): # if has multiple GPUs
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous) # call super-resolution function, with the low-resolution but interpolated image in data['SR']
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train() # set the diffusion model back to train mode

    def sample(self, batch_size=1, continous=False):
        """
        Used to generate random images from the diffusion model.
        """

        self.netG.eval() # set the diffusion model to evaluation mode
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous) # generate samples with batch_size
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        """
        Set a new noise schedule for the Diffusion model
        schedule_opt: a dictionary containing the options for the noise schedule
        schedule_phase: whether it is train or val
        """
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            # update the schedule phase to provided phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)
            # call the "set_new_noise_schedule" method of the generator network

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        """
        Collect and prepare the relevant images generated during the testing process.
        Returns an ordered dictionary with keys and image tensors.
        """
        out_dict = OrderedDict()
        if sample: # if the sample flag is true, add the sample images
            out_dict['SAM'] = self.SR.detach().float().cpu() # detached from computation graph and move to CPU
        else: # if sample is false, the function retrieves images related to training/testing process
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
