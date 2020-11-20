import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

from navrep.models.torchvae import VAE1D
from navrep.models.gpt import Block

_A = 3
_G = 2
_H = 64
_Z = _H

logger = logging.getLogger(__name__)

STATE_NORM_FACTOR = 25.  # maximum typical goal distance, meters

class GPT1D(nn.Module):
    def __init__(self, config, gpu=True):
        super().__init__()

        # input embedding stem
        self.convVAE = VAE1D(z_dim=config.n_embd, gpu=gpu)
        self.action_emb = nn.Linear(config.n_action, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.state_emb = nn.Linear(config.n_states, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.z_head = nn.Linear(config.n_embd, config.n_embd)
        self.state_head = nn.Linear(config.n_embd, config.n_states)

        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.apply(self._init_weights)

        self.gpu = gpu

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, lidar, state, action, dones, targets=None, h=None):
        """
        lidar: (batch, sequence, CH, L) [0, 1]
        action: (batch, sequence, A) [-inf, inf]
        state: (batch, sequence, S) [-inf, inf]
        dones:  (batch, sequence,) {0, 1}
        targets: None or (lidar_targets, state_targets)
            lidar_targets: same shape as lidar
            state_targets: same shape as state

        OUTPUTS
        lidar_pred: same shape as lidar
        state_pred: same shape as state
        loss: torch loss
        """
        b, t, CH, L = lidar.size()
        _, _, A = action.size()
        _, _, S = state.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # encode embedding with vae
        z, mu, logvar = self.convVAE.encode(lidar.view(b * t, CH, L))  # each image maps to a vector
        token_embeddings = z.view(b, t, self.n_embd)
        state_embeddings = self.state_emb(state.view(b * t, S)).view(b, t, self.n_embd)
        action_embeddings = self.action_emb(action.view(b * t, A)).view(b, t, self.n_embd)
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        # forward the GPT model
        x = self.drop(token_embeddings + position_embeddings + action_embeddings + state_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        # store worldmodel embedding
        if h is not None:
            h[0] = x
        # decode embedding with vae
        z_pred = self.z_head(x.view(b * t, self.n_embd))
        lidar_rec = self.convVAE.decode(z).view(b, t, CH, L)
        lidar_pred = self.convVAE.decode(z_pred).view(b, t, CH, L)
        state_pred = self.state_head(x.view(b * t, self.n_embd)).view(b, t, S)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            lidar_targets, state_targets = targets
            rec_loss = F.mse_loss(lidar_rec, lidar)  # input-reconstruction loss
            pred_loss = F.mse_loss(lidar_pred, lidar_targets)  # reconstructed prediction loss
            state_loss = F.mse_loss(state_pred, state_targets) / STATE_NORM_FACTOR**2
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # kullback leibler
            kld_tolerance = 0.5
            kld_weight = 0.001
            KLD = torch.max(KLD, kld_tolerance * torch.ones_like(KLD))
            loss = rec_loss + kld_weight * KLD + pred_loss + state_loss

        return lidar_pred, state_pred, loss

    def _to_correct_device(self, tensor):
        if self.gpu:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                return tensor.to(device)
            else:
                print("WARNING: model created with gpu enabled, but no gpu found")
        return tensor

    def encode(self, lidar):
        """
        lidar: numpy (batch, L, CH)

        OUTPUTS
        z: (batch, Z)
        """
        b, L, CH = lidar.shape

        lidar_t = torch.tensor(np.moveaxis(lidar, -1, 1), dtype=torch.float)
        lidar_t = self._to_correct_device(lidar_t)

        z, mu, logvar = self.convVAE.encode(lidar_t)
        return z.detach().cpu().numpy()

    def encode_mu_logvar(self, lidar):
        """
        lidar: numpy (batch, L, CH)


        OUTPUTS
        mu: (batch, Z)
        logvar: (batch, Z)
        """
        b, L, CH = lidar.shape

        lidar_t = torch.tensor(np.moveaxis(lidar, -1, 1), dtype=torch.float)
        lidar_t = self._to_correct_device(lidar_t)

        z, mu, logvar = self.convVAE.encode(lidar_t)
        mu = mu.detach().cpu().numpy()
        logvar = logvar.detach().cpu().numpy()
        return mu, logvar

    def decode(self, z):
        """
        z: numpy (batch, Z)

        OUTPUTS
        lidar_rec: (batch, L, CH)
        """
        b, Z = z.shape

        z_t = torch.tensor(z, dtype=torch.float)
        z_t = self._to_correct_device(z_t)

        lidar_rec_t = self.convVAE.decode(z_t) # b, CH, L
        lidar_rec = np.moveaxis(lidar_rec_t.detach().cpu().numpy(), 1, -1)
        return lidar_rec

    def get_h(self, gpt_sequence):
        """ for compat with encodedenv
        gpt sequence is a list of dicts, one for each step in the sequence.
        each dict has
        "obs": numpy (L, CH) [0, 1]
        "state": numpy (2,) [-inf, inf]
        "action": numpy (3,) [-inf, inf]
        """
        _b = 1  # batch size
        lidar = np.array([d["obs"] for d in gpt_sequence])  # t, L, CH
        lidar = np.moveaxis(lidar, -1, 1)
        lidar = lidar.reshape((_b, *lidar.shape))
        lidar_t = torch.tensor(lidar, dtype=torch.float)
        lidar_t = self._to_correct_device(lidar_t)
        state = np.array([d["state"] for d in gpt_sequence])  # t, 2
        state = state.reshape((_b, *state.shape))
        state_t = torch.tensor(state, dtype=torch.float)
        state_t = self._to_correct_device(state_t)
        action = np.array([d["action"] for d in gpt_sequence])  # t, 3
        action = action.reshape((_b, *action.shape))
        action_t = torch.tensor(action, dtype=torch.float)
        action_t = self._to_correct_device(action_t)
        dones = np.zeros((_b, len(gpt_sequence), 1))
        dones_t = torch.tensor(dones, dtype=torch.float)
        dones_t = self._to_correct_device(dones_t)
        h_container = [None]
        self.forward(lidar_t, state_t, action_t, dones_t, h=h_container)
        h = h_container[0].detach().cpu().numpy()
        h = h[0, -1]  # only batch, last item in sequence
        return h
