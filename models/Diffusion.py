import numpy as np
import torch

class DiffusionModel(torch.nn.Module):
    def __init__(self, model, betas, loss, device, clip: bool = False) -> None:
        super().__init__()
        
        # Assume 'self' is alreay on the proper device
        # device held because the class does some generation
        self.device = device
        self.model = model

        self.loss = loss
        self.sample_clip = clip
        
        assert self.loss in ["mse", "mae"]
        assert (betas > 0.).all() and (betas <= 1.).all()

        self.register_buffer("trained_betas", torch.tensor(betas, dtype=torch.float32))
        self.set_variances(np.arange(0, betas.shape[0]), True)

    def reset_timesteps(self):
        self.set_num_timesteps(self.trained_betas.shape[0])

    def set_num_timesteps(self, T): 
        T = min(T, self.trained_betas.shape[0])
        self.set_subsequence(np.linspace(0, self.trained_betas.shape[0], T, endpoint=False).astype(int))

    def set_subsequence(self, subsequence = None):
        if subsequence is None:
            subsequence = np.arange(0, self.betas_base.shape[0])
        self.set_variances(subsequence)

    def set_variances(self, subsquence, create=False):
        if create:
            def register(name, value):
                self.register_buffer(name, value)
        else:
            def register(name, value):
                param = getattr(self, name)
                param.data = value.data
        
        betas = self.trained_betas[subsquence]

        self.num_timesteps = betas.shape[0]
        register("betas", betas)
        alphas = 1. - self.betas 

        # Register as buffer for device to apply automatically
        register("alphas_cumprod", torch.cumprod(alphas, 0))
        register("alphas_cumprod_prev", torch.cat([torch.Tensor([1]).to(self.alphas_cumprod.device), self.alphas_cumprod[:-1]]))

        # Computation for diffusion q(x_t | x_{t-1})
        register("sqrt_alphas_cumprod",           torch.sqrt(self.alphas_cumprod))
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - self.alphas_cumprod))
        register("log_one_minus_alphas_cumprod" , torch.log (1. - self.alphas_cumprod))
        register("sqrt_recip_alphas_cumprod"    , torch.sqrt(1. / self.alphas_cumprod))
        register("sqrt_recipm1_alphas_cumprod"  , torch.sqrt(1. / self.alphas_cumprod - 1.))

        # Computation for posterior q(x_{t-1} | x_t, x_0)
        register("posterior_variance", self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        register("posterior_log_variance_clipped", torch.log(
            torch.cat([
                torch.Tensor([self.posterior_variance[1]]).to(self.posterior_variance.device), 
                self.posterior_variance[1:]
            ]))
        )
        register("posterior_mean_coef1", self.betas * torch.sqrt(self.alphas_cumprod_prev)   / (1 - self.alphas_cumprod))
        register("posterior_mean_coef2", (1 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - self.alphas_cumprod))

        # 
        self.to(self.device)

    @staticmethod
    def extract(data, at, shape):
        bs = shape[0]
        
        out = torch.gather(data, 0, at)
        # Reshape for broadcasting purposes
        return out.view([bs] + (len(shape) - 1) * [1])

    def noise_fn(self, shape):
        return torch.randn(shape, dtype=torch.float32).to(self.device)

    def q_mean_variance(self, x_start, t):
        mean = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        var = self.extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_var = self.extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, var, log_var

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.noise_fn(x_start.shape)

        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_xstart_from_noise(self, x_t, t, noise):
        assert x_t.shape == noise.shape
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            self.extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            self.extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )

    def p_mean_variance(self, x, cond, t, clip_denoised: bool, return_pred_xstart: bool):
        # B, H, W, C = x.shape
        predictions = self.model(x, t, cond)

        model_mean = predictions
        model_var = self.betas
        model_logvar = torch.log(
            torch.cat([
                torch.Tensor([self.posterior_variance[1]]).to(self.device), 
                self.betas[1:]
            ])
        )
        model_var = self.extract(model_var, t, x.shape) * torch.ones_like(x)
        model_logvar = self.extract(model_logvar, t, x.shape) * torch.ones_like(x)
        
        # Clip only when asked to
        clip_denoised = False
        clip = (lambda x_: torch.clip(x_, -1, 1)) if clip_denoised else (lambda x_: x_)
        pred_xstart = clip(self.predict_xstart_from_noise(x, t, predictions))
        model_mean, _, _, = self.q_posterior_mean_variance(pred_xstart, x, t)
    
        if return_pred_xstart:
            return model_mean, model_var, model_logvar, pred_xstart
        
        return model_mean, model_var, model_logvar
    
    # Sampling
    def p_sample(self, x, cond, t, clip_denoised: bool = True, return_pred_xstart: bool = False, with_sampling: bool = True):
        model_mean, _, model_logvar, pred_xstart = self.p_mean_variance(
            x, cond, t, clip_denoised, True
        )
        if with_sampling:
            noise = self.noise_fn(x.shape).to(x.dtype)
            t = t.view(t.shape[0], *[1] * (noise.ndim - 1))
            noise = torch.where(t == 0, 0, noise)
            sample = model_mean + torch.exp(0.5 * model_logvar) * noise
        else:
            sample = model_mean

        if return_pred_xstart:
            return sample, pred_xstart
        
        return sample
    
    def p_sample_loop(self, shape, img = None, cond = None, with_tqdm=True, with_sampling=True):
        if img is None:
            img = self.noise_fn(shape)
        else:
            shape = img.shape
        img = img.to(self.device)

        if cond is not None:
            cond = cond.to(self.device)

        # if with_tqdm and True:
        if True:
            from tqdm import tqdm
            iterator = tqdm(reversed(range(self.num_timesteps - 1)), total=self.num_timesteps-1)
        else:
            iterator = reversed(range(self.num_timesteps - 1))

        for i in iterator:
            # print(i, self.num_timesteps, shape)
            t = int(i) * torch.ones(shape[0], dtype=torch.int64).to(self.device)
            img = self.p_sample(img, cond, t, clip_denoised=self.sample_clip, return_pred_xstart=False, with_sampling=with_sampling)
        
        return img

    def forward(self, x_start, t, noise=None, x_cond=None):
        assert t.shape[0] == x_start.shape[0]
        
        if noise is None:
            noise = self.noise_fn(x_start.shape)
        
        # Add noise to data
        x_t = self.q_sample(x_start, t, noise)
        predictions = self.model(x_t, t, x_cond)
        
        target = noise
        errs = (predictions - target) 

        if self.loss == "mse":
            errs = torch.square(errs)
        elif self.loss == "mae":
            errs = torch.abs(errs)
        else:
            print("Unknown loss !!")

        loss = errs.view(errs.shape[0], -1).mean(1)
        return loss.mean()