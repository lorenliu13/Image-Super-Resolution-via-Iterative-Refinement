import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # config: set up a JSON file for configuration, two alternative names "-c" and "--config"
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_era_32_128_local_test.json',
                        help='JSON file for configuration')
    # phase: whether the script should be run in training or validation (val) mode
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    # gpus_ids: specify the GPU device, None means use available GPUs
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    # debug: enable debug mode for the script
    parser.add_argument('-debug', '-d', action='store_true')
    # enable_wandb: enable the use of Weights and Biases for experiment tracking and visualization
    parser.add_argument('-enable_wandb', action='store_true')
    # log_wandb_ckpt: enables checkpoints in W&B
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    # log_eval: enable logging evaluation results during training
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True # enable cuDNN library to improve speed
    torch.backends.cudnn.benchmark = True # enable cuDNN benchmark mode, improve performance

    # set up a logger for the training phrase
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    # set up a logger for the validation phrase
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)

    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt) # create a new WandbLogger instance
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items(): # loop through the 'datasets' dictionary
        if phase == 'train' and args.phase != 'val': # if the current phrase is 'train'
            train_set = Data.create_dataset(dataset_opt, phase) # create dataset
            train_loader = Data.create_dataloader( # create a data loader
                train_set, dataset_opt, phase)
        elif phase == 'val': # if the phrase is validation
            val_set = Data.create_dataset(dataset_opt, phase) # create a dataset object
            val_loader = Data.create_dataloader( # create a data loader object
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished') # logs an informational message

    # model
    diffusion = Model.create_model(opt) # create an instance of the diffusion model
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step # retrieve the starting step of the model
    current_epoch = diffusion.begin_epoch # retrieve the starting epoch of the model
    n_iter = opt['train']['n_iter'] # retrieve the number of training iterations

    if opt['path']['resume_state']: # check if "resume_state" is specified
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step)) # if so, showing that is being resumed

    # set the noise schedule for the diffusion process
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train': # if in train phrase
        while current_step < n_iter: # when still within n_iter
            current_epoch += 1
            for _, train_data in enumerate(train_loader): # iterate through training data loader, loading batches of training data
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data) # feed the data into the diffusion model
                diffusion.optimize_parameters() # update model's parameters
                # log
                # log training metrics at intervals
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                # calculate PSNR (peak signal-to-noise ratio) over validation dataset
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch) # create the result path "results/current_epoch"
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    # for each image in the validation set
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        # convert the tensor to numpy array
                        sr_img = Metrics.tensor2numpy(visuals['SR']) # (128, 128) between 0 and 1
                        hr_img = Metrics.tensor2numpy(visuals['HR']) # (128, 128)
                        lr_img = Metrics.tensor2numpy(visuals['LR']) # (128, 128)
                        fake_img = Metrics.tensor2numpy(visuals['INF']) # (128, 128)

                        # generation
                        Metrics.save_numpy(
                            hr_img, '{}/{}_{}_hr.npy'.format(result_path, current_step, idx))
                        Metrics.save_numpy(
                            sr_img, '{}/{}_{}_sr.npy'.format(result_path, current_step, idx))
                        Metrics.save_numpy(
                            lr_img, '{}/{}_{}_lr.npy'.format(result_path, current_step, idx))
                        Metrics.save_numpy(
                            fake_img, '{}/{}_{}_inf.npy'.format(result_path, current_step, idx))

                        # expand one dimension to (1, 128, 128)
                        fake_img_expanded = fake_img[np.newaxis, :, :]
                        sr_img_expanded = sr_img[np.newaxis, :, :]
                        hr_img_expanded = hr_img[np.newaxis, :, :]

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.concatenate(
                                (fake_img_expanded, sr_img_expanded, hr_img_expanded), axis=2),
                            idx)
                            # concatenate to be (1, 128, 384)

                        avg_psnr += Metrics.calculate_psnr_npy(
                            sr_img, hr_img) # calculate psnr, given that image range from 0 to 1

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img_expanded, sr_img_expanded, hr_img_expanded), axis=2)
                            ) # concatenate to be (1, 128, 384)

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                # save the model and training states
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)
                    # checkpoints are also logged to Weights & Biases
                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)
            # logs the current epoch to Weights & Biases
            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader): # iterates through the validation data loader
            idx += 1
            diffusion.feed_data(val_data) # feed the validation data to the diffusion model
            diffusion.test(continous=True) # runs the model in test mode, generating super-resolution images from LR images
            visuals = diffusion.get_current_visuals() # retrieves the current visuals, including LR, HR, and generated imgs

            hr_img = Metrics.tensor2numpy(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2numpy(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2numpy(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_numpy(
                        Metrics.tensor2numpy(sr_img[iter]), '{}/{}_{}_sr_{}.npy'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2numpy(visuals['SR'])  # uint8
                Metrics.save_numpy(
                    sr_img, '{}/{}_{}_sr_process.npy'.format(result_path, current_step, idx))
                Metrics.save_numpy(
                    Metrics.tensor2numpy(visuals['SR'][-1]), '{}/{}_{}_sr.npy'.format(result_path, current_step, idx))

            # save imgs
            Metrics.save_numpy(
                hr_img, '{}/{}_{}_hr.npy'.format(result_path, current_step, idx))
            Metrics.save_numpy(
                lr_img, '{}/{}_{}_lr.npy'.format(result_path, current_step, idx))
            Metrics.save_numpy(
                fake_img, '{}/{}_{}_inf.npy'.format(result_path, current_step, idx))

            # generation
            # calculates PNSR and SSIM between HR and generated super-resolution imgs
            eval_psnr = Metrics.calculate_psnr_npy(Metrics.tensor2numpy(visuals['SR'][-1]), hr_img)

            # convert the image from [0,1] to [0, 255] before calculate the ssim
            eval_ssim = Metrics.calculate_ssim((Metrics.tensor2numpy(visuals['SR'][-1]) * 255).astype(np.uint8), (hr_img * 255).astype(np.uint8))

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2numpy(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
