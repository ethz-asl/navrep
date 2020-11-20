#!/usr/bin/env bats

@test "check that navrep folder is clean" {
  [ -d "~/navrep/models" ]
  [ -d "~/navrep/logs" ]
  [ -d "~/navrep/eval" ]
}

@test "make vae dataset" {
  ipython -- make_vae_dataset.py --environment navreptrain --n 3
}

# VAE1D_LSTM
@test "train vae1d" {
  ipython -- train_vae1d.py --environment navreptrain --n 1000
}
@test "make 1d rnn dataset" {
  ipython -- make_1drnn_dataset.py --environment navreptrain --n 3
}
@test "train 1d rnn" {
  ipython -- train_rnn.py --environment navreptrain --backend VAE1D_LSTM --n 60
}
@test "gym vae1d_lstm v_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAE1D_LSTM --encoding V_ONLY --no-gpu --n 100000
}
@test "gym vae1d_lstm m_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAE1D_LSTM --encoding M_ONLY --no-gpu --n 100000
}
@test "gym vae1d_lstm vm" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAE1D_LSTM --encoding VM --no-gpu --n 100000
}

# VAE_LSTM
@test "train vae" {
  ipython -- train_vae.py --environment navreptrain --n 1000
}
@test "make rnn dataset" {
  ipython -- make_rnn_dataset.py --environment navreptrain --n 3
}
@test "train rnn" {
  ipython -- train_rnn.py --environment navreptrain --n 60
}
@test "gym vae_lstm v_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAE_LSTM --encoding V_ONLY --no-gpu --n 100000
}
@test "gym vae_lstm m_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAE_LSTM --encoding M_ONLY --no-gpu --n 100000
}
@test "gym vae_lstm vm" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAE_LSTM --encoding VM --no-gpu --n 100000
}

# VAELSTM
@test "train vaelstm" {
  ipython -- train_vaelstm.py --environment navreptrain --n 100
}
@test "gym vaelstm v_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAELSTM --encoding V_ONLY --n 100000
}
@test "gym vaelstm m_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAELSTM --encoding M_ONLY --n 100000
}
@test "gym vaelstm vm" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAELSTM --encoding VM --n 100000
}

# VAE1DLSTM
@test "train vae1dlstm" {
  ipython -- train_vae1dlstm.py --environment navreptrain --n 100
}
@test "gym vae1dlstm v_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAE1DLSTM --encoding V_ONLY --n 100000
}
@test "gym vae1dlstm m_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAE1DLSTM --encoding M_ONLY --n 100000
}
@test "gym vae1dlstm vm" {
  ipython -- train_gym_navreptrainencodedenv.py --backend VAE1DLSTM --encoding VM --n 100000
}

# GPT
@test "train gpt" {
  ipython -- train_gpt.py --environment navreptrain --n 100
}
@test "gym gpt v_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend GPT --encoding V_ONLY --n 100000
}
@test "gym gpt m_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend GPT --encoding M_ONLY --n 100000
}
@test "gym gpt vm" {
  ipython -- train_gym_navreptrainencodedenv.py --backend GPT --encoding VM --n 100000
}

# GPT 1D
@test "train gpt1d" {
  ipython -- train_gpt1d.py --environment navreptrain --n 100
}
@test "gym gpt1d v_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend GPT1D --encoding V_ONLY --n 100000
}
@test "gym gpt1d m_only" {
  ipython -- train_gym_navreptrainencodedenv.py --backend GPT1D --encoding M_ONLY --n 100000
}
@test "gym gpt1d vm" {
  ipython -- train_gym_navreptrainencodedenv.py --backend GPT1D --encoding VM --n 100000
}
