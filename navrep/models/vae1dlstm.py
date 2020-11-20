import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from navrep.models.torchvae import VAE1D

_A = 3
_G = 2

STATE_NORM_FACTOR = 25.  # maximum typical goal distance, meters

class VAEConfig:
    def __init__(self, z_size):
        self.z_size = z_size

class LSTMConfig:
    n_layer = 2

    def __init__(self, h_size):
        self.h_size = h_size

class VAE1DLSTMConfig:
    pdrop = 0.1

    n_action = _A  # forward v, side v, rot v
    n_states = _G  # goal_x, goal_y (in robot frame)

    def __init__(self, z_size, h_size):
        self.vae = VAEConfig(z_size)
        self.lstm = LSTMConfig(h_size)

class VAE1DLSTM(nn.Module):
    def __init__(self, config, gpu=True):
        super().__init__()

        # input embedding stem
        self.convVAE = VAE1D(z_dim=config.vae.z_size, gpu=gpu)
        self.action_emb = nn.Linear(config.n_action, config.vae.z_size)
        self.state_emb = nn.Linear(config.n_states, config.vae.z_size)
        self.drop = nn.Dropout(config.pdrop)
        # transformer
        self.lstm = nn.LSTM(config.vae.z_size * 3,  # *3 due to concat (z, action, state)
                            config.lstm.h_size, config.lstm.n_layer,
                            batch_first=True)
        self.lstm_head = nn.Linear(config.lstm.h_size, config.vae.z_size * 2) # *2 (no action pred)
        # decoder head
        self.ln_f = nn.LayerNorm(config.vae.z_size * 2)
        self.z_head = nn.Linear(config.vae.z_size, config.vae.z_size)
        self.state_head = nn.Linear(config.vae.z_size, config.n_states)

        self.config = config
        self.gpu = gpu

        self.apply(self._init_weights)

        print(
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

    def forward(self, lidar, state, action, dones, rnn_state, targets=None):
        """
        lidar: (batch, sequence, CH, L) [0, 1]
        action: (batch, sequence, A) [-inf, inf]
        state: (batch, sequence, S) [-inf, inf]
        dones:  (batch, sequence,) {0, 1}
        targets: None or (lidar_targets, state_targets)
            lidar_targets: same shape as lidar
            state_targets: same shape as state
        rnn_state: [h0, c0]

        OUTPUTS
        lidar_pred: same shape as lidar
        state_pred: same shape as state
        loss: torch loss
        rnn_state: contents get updated with hn, cn
        """
        b, t, CH, L = lidar.size()
        _, _, A = action.size()
        _, _, S = state.size()

        h0 = rnn_state[0]
        c0 = rnn_state[1]

        # encode embedding with vae
        z, mu, logvar = self.convVAE.encode(
            lidar.view(b * t, CH, L)
        )  # each image maps to a vector
        z_embeddings = z.view(b, t, self.config.vae.z_size)
        state_embeddings = self.state_emb(state.view(b * t, S)).view(b, t, self.config.vae.z_size)
        action_embeddings = self.action_emb(action.view(b * t, A)).view(
            b, t, self.config.vae.z_size
        )
        # forward the prediction model
        x = self.drop(torch.cat([z_embeddings, action_embeddings, state_embeddings], dim=-1))
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = x.reshape(b * t, self.config.lstm.h_size) # view() doesn't work...
        x = self.lstm_head(x)
        x = self.ln_f(x)
        z_embedding_pred = x[:, :self.config.vae.z_size]
        state_embedding_pred = x[:, self.config.vae.z_size:]
        # store final rnn state
        if rnn_state is not None:
            rnn_state[0] = hn
            rnn_state[1] = cn
        # decode embedding with vae
        state_pred = self.state_head(state_embedding_pred).view(b, t, S)
        z_pred = self.z_head(z_embedding_pred)
        lidar_rec = self.convVAE.decode(z).view(b, t, CH, L)
        lidar_pred = self.convVAE.decode(z_pred).view(b, t, CH, L)

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
        "obs": numpy lidar (L, CH) [0, 1]
        "state": numpy (2,) [-inf, inf]
        "action": numpy (3,) [-inf, inf]

        h: numpy (h_size,)  the hidden state at the end of the sequence
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
        h0 = torch.randn(self.config.lstm.n_layer, _b, self.config.lstm.h_size)
        c0 = torch.randn(self.config.lstm.n_layer, _b, self.config.lstm.h_size)
        h0 = self._to_correct_device(h0)
        c0 = self._to_correct_device(c0)
        rnn_state_container = [h0, c0]
        self.forward(lidar_t, state_t, action_t, dones_t, rnn_state=rnn_state_container)
        h = rnn_state_container[0]
        h = h.view(self.config.lstm.n_layer, _b, self.config.lstm.h_size)
        h = h.detach().cpu().numpy()
        h = h[-1, 0, :]  # last layer, only batch
        return h
