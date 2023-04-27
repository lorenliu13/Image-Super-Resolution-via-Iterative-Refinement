import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    Return an array of beta values that define the noise schedule for the Gaussian diffusion process.
    This noise addition process is controlled by a set of parameters called "betas."
    beta = 1 - alpha

    """

    if schedule == 'quad':
        # Quadratic schedule (quad): The betas increase quadratically from a specified start to a specified end value.
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        # Linear schedule (linear): The betas increase linearly from a specified start to a specified end value.
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        # Warmup schedules (warmup10, warmup50): The betas are set to gradually increase in the beginning before transitioning to the main schedule.
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        # Constant schedule (const): The betas remain constant throughout the diffusion process.
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # JSD schedule (jsd): The betas are set using the inverse of the timesteps.
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        # Cosine schedule (cosine): The betas follow a cosine function that starts at a specified value and oscillates.
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        """
        denoise_fn: the denoising function (U-Net) used in the diffusion process
        image_size: the size of the image being processed
        channels: the number of channels in the images
        lss_type: the type of loss function
        conditional: a boolean flag indicating if the diffusion process is conditional on some other input
        schedule_opt: specify a noise schedule for the diffusion process
        """

        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        """
        Create and set a noise schedule for the Gaussian Diffusion process
        schedule_opt: A dictionary containing the noise schedule configuration
        device: the device where the tensors should be allocated
        """

        # create a partial function that converts a numpy array to a pytorch tensor and allocate it on the device
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # calculate betas
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas # retrieve to cpu as numpy array
        # calculate alphas
        alphas = 1. - betas
        # calculate cumulative product of alpha, i.e., gammas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # get an array that saves gamma at the previous time steps
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # calculate the square root of the cumulative product of alphas,
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))
        # extract the number of time steps
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # register various calculated numpy arrays
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # equation 4
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20)))) # calculate log(variance)
        # calculate the mean of posterior distribution
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Compute the predicted starting point of the diffusion process (i.e., x0) given the current image x_t and noise
        xt: the noisy image, i.e., yt in the paper
        t: the current time step
        noise: the forward noise "learned" by the U-net
        return: the estimated start image y0
        """
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise # equation 10

    def q_posterior(self, x_start, x_t, t):
        """
        This is for forward noise process q
        Compute the posterior mean and log variance of the distribution q(x t-1 | xt, x0) at given time step t
        x_start: the first image without any noise, i.e., y0
        xt: the current noisy image, i.e., yt
        t: the current time step
        return: the mean and variance of p(x_{t-1} | x_t, x_0)
        """
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t # compute the posterior mean (equation 4)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t] # log variance (equation 4)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        """
        Compute the backward denoise process p
        Compute the model mean and posterior log variance of the distribution p(x_{t-1} | x_t, x_0)
        x: currenty noisy image yt at time step t
        t: time step
        clip_denoised: a boolean indicating whether to clamp the denoised image
        condition_x: if conditioned on x
        return: the mean and variance of p(x_{t-1} | x_t, x_0)
        """
        batch_size = x.shape[0] # the size of the batch from the input x
        # repeating the sqrt of gamma to batch_size
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            # if condition_x is provided, concatenate it with x along the channel dimension
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
            # predict_start_from_noise: to compute denoised image x_recon
            # denoise_fn: denoising function returns a noise tensor
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))
            # predict_start_from_noise(x_t = x, t = t, noise = denoise_fn(x, noise_level))
        # if clip_denoised is true, clamp the denoised image between -1 and 1
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        # compute the mean and log variance
        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        # q_posterior(x_start = x_recon, x_t = xt, t = t)
        return model_mean, posterior_log_variance

    @torch.no_grad() # indicate the function should not track gradients
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        """
        Generate a sample from the distribution p(x_{t-1} | x_t, x_0)
        x: the current noisy image yt at time step t
        t: the current time step
        condition_x: the low resolution image
        return: the new image yt-1 generated at t-1
        """
        # Calculate the model mean and log variance of the denoising step
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        # generate random noise with the same shape as x, if t > 0
        # if t = 0, create a tensor of zeros, this is because no noise at the initial time step
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        # compute the final noise from the distribution p(x_{t-1} | x_t, x_0)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        """
        Generate a sequence of samples during the reverse diffusion process.
        """
        # get the device information
        device = self.betas.device
        # determine the interval at whcih images are collected during the sampling process
        # the interval is either 1 or (num_timsteps//10)
        sample_inter = (1 | (self.num_timesteps//10))
        # if the model is unconditional
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device) # set image as a random noise tensor
            ret_img = img
            # iterate the time steps in reverse order
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                # for a time step
                img = self.p_sample(img, i) # generate random samples from p and update img
                if i % sample_inter == 0:
                    # if at the interval, concat the generated sample to ret_img along the first dimension
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            # if the model is conditional
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device) # set img to a random noise
            ret_img = x # set ret_img to x
            # Iterate through the time steps in reverse order.
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):

                img = self.p_sample(img, i, condition_x=x) # generate a sample with x as conditions
                if i % sample_inter == 0:
                    # if at the save interval, concat the generated sample to rec_img
                    ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            # if continous is true, return ret_img that contains the entire sequence of generated images
            return ret_img
        else:
            # otherwise, only return the last image in the sequence
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        """
        Used to generate new samples using the diffusion model
        batch_size: can use batch size
        """
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        """
        Generate super-resolution images using conditional diffusion model
        x_in: low-resolution images
        continous: whether to save internal denoising images
        """
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        """
        Add noise to original images (high-resolution)
        x_start: the starting image
        continuous_sqrt_alpha_cumprod: square root of the cumulative product of alphas
        noise:
        """
        # if noise is not provided, it generates rnadom noise with the same shape as x_start
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        # add noise to original images
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        """
        x_in: a dictionary containing HR and SR images
        noise: optional argument
        """

        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1) # select a random timestep t
        # get the sqrt of cumulative product of alpha, using a piece wise distribution
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        # generate a noisy image using q_sample method
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        # if the model is not conditional
        if not self.conditional:
            # get reconstructed noise using denoise_fn
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            # if it is conditional, get reconstructed noise using noisy image and SR image
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
        # compute the loss between generated noise and the reconstructed noise
        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
