'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''
    create dataloader
    dataset: an instance of the Dataset class
    dataset_opt: configuration setting
    phrase: whether is train or val
    '''

    if phase == 'train':
        # create a data loader with certain batch_size, use_shuffle, and num_workers
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        # create a data loader with batch size of 1, no shuffling, and a single worker
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    # from data.LRHR_dataset import LRHRDataset as D
    from data.ERA5_LRHR_npy_dataset import ERA5_LRHRDataset as D # import dataset support ERA5
    # call data.LRHR_dataset
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR') # if mode == 'LRHR', true.
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
