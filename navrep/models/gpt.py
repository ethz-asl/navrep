"""
Transformer blocks taken from Karpathy's minGPT (MIT License) https://github.com/karpathy/minGPT
"""

import random
import numpy as np
import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

from navrep.models.torchvae import VAE

logger = logging.getLogger(__name__)

_A = 3
_G = 2

STATE_NORM_FACTOR = 25.  # maximum typical goal distance, meters

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    n_layer = 8
    n_head = 8
    n_action = _A  # forward v, side v, rot v
    n_states = _G  # goal_x, goal_y (in robot frame)

    def __init__(self, block_size, n_embd):
        self.block_size = block_size
        self.n_embd = n_embd  # good size is 64

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(
            self.mask[:, :, :T, :T] == 0, -1e10
        )  # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """ A prediction model based on openAI's GPT architecture """

    def __init__(self, config, gpu=True):
        super().__init__()

        # input embedding stem
        self.convVAE = VAE(z_dim=config.n_embd, gpu=gpu)
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

    def forward(self, img, state, action, dones, targets=None, h=None):
        """
        img: (batch, sequence, CH, W, H) [0, 1]
        action: (batch, sequence, A) [-inf, inf]
        state: (batch, sequence, S) [-inf, inf]
        dones:  (batch, sequence,) {0, 1}
        targets: None or (img_targets, state_targets)
            img_targets: same shape as img
            state_targets: same shape as state

        OUTPUTS
        img_pred: same shape as img
        state_pred: same shape as state
        loss: torch loss
        """
        b, t, CH, W, H = img.size()
        _, _, A = action.size()
        _, _, S = state.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # encode embedding with vae
        z, mu, logvar = self.convVAE.encode(img.view(b * t, CH, W, H))  # each image maps to a vector
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
        img_rec = self.convVAE.decode(z).view(b, t, CH, W, H)
        img_pred = self.convVAE.decode(z_pred).view(b, t, CH, W, H)
        state_pred = self.state_head(x.view(b * t, self.n_embd)).view(b, t, S)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            img_targets, state_targets = targets
            rec_loss = F.binary_cross_entropy(img_rec, img)  # input-reconstruction loss
            pred_loss = F.binary_cross_entropy(img_pred, img_targets)  # reconstructed prediction loss
            state_loss = F.mse_loss(state_pred, state_targets) / STATE_NORM_FACTOR**2
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # kullback leibler
            kld_tolerance = 0.5
            kld_weight = 0.001
            KLD = torch.max(KLD, kld_tolerance * torch.ones_like(KLD))
            loss = rec_loss + kld_weight * KLD + pred_loss + state_loss

        return img_pred, state_pred, loss

    def _to_correct_device(self, tensor):
        if self.gpu:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                return tensor.to(device)
            else:
                print("WARNING: model created with gpu enabled, but no gpu found")
        return tensor

    def encode(self, img):
        """
        img: numpy (batch, W, H, CH)

        OUTPUTS
        z: (batch, Z)
        """
        b, W, H, CH = img.shape

        img_t = torch.tensor(np.moveaxis(img, -1, 1), dtype=torch.float)
        img_t = self._to_correct_device(img_t)

        z, mu, logvar = self.convVAE.encode(img_t)
        return z.detach().cpu().numpy()

    def encode_mu_logvar(self, img):
        """
        img: numpy (batch, W, H, CH)


        OUTPUTS
        mu: (batch, Z)
        logvar: (batch, Z)
        """
        b, W, H, CH = img.shape

        img_t = torch.tensor(np.moveaxis(img, -1, 1), dtype=torch.float)
        img_t = self._to_correct_device(img_t)

        z, mu, logvar = self.convVAE.encode(img_t)
        mu = mu.detach().cpu().numpy()
        logvar = logvar.detach().cpu().numpy()
        return mu, logvar

    def decode(self, z):
        """
        z: numpy (batch, Z)

        OUTPUTS
        img_rec: (batch, W, H, CH)
        """
        b, Z = z.shape

        z_t = torch.tensor(z, dtype=torch.float)
        z_t = self._to_correct_device(z_t)

        img_rec_t = self.convVAE.decode(z_t) # b, CH, W, H
        img_rec = np.moveaxis(img_rec_t.detach().cpu().numpy(), 1, -1)
        return img_rec

    def get_h(self, gpt_sequence):
        """ for compat with encodedenv
        gpt sequence is a list of dicts, one for each step in the sequence.
        each dict has
        "obs": numpy image (W, H, CH) [0, 1]
        "state": numpy (2,) [-inf, inf]
        "action": numpy (3,) [-inf, inf]
        """
        _b = 1  # batch size
        img = np.array([d["obs"] for d in gpt_sequence])  # t, W, H, CH
        img = np.moveaxis(img, -1, 1)
        img = img.reshape((_b, *img.shape))
        img_t = torch.tensor(img, dtype=torch.float)
        img_t = self._to_correct_device(img_t)
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
        self.forward(img_t, state_t, action_t, dones_t, h=h_container)
        h = h_container[0].detach().cpu().numpy()
        h = h[0, -1]  # only batch, last item in sequence
        return h

def save_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        ckpt_model = model.module if hasattr(model, "module") else model
        logger.info("saving %s", ckpt_path)
        torch.save(ckpt_model.state_dict(), ckpt_path)

def load_checkpoint(model, ckpt_path, gpu=False):
    if gpu:
        device = torch.cuda.current_device()
        map_location = "cuda:{}".format(device)
    else:
        map_location = torch.device('cpu')
    state_dict = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state_dict)
    if gpu:
        model.to(device)
