import pytest
import os

from pyniel.python_tools.bash import bash


@pytest.mark.dependency()
def test_navrep_is_clean():
    assert not os.path.isdir(os.path.expanduser("~/navrep/models"))
    assert not os.path.isdir(os.path.expanduser("~/navrep/logs"))
    assert not os.path.isdir(os.path.expanduser("~/navrep/eval"))

@pytest.mark.dependency()
def test_make_vae_dataset():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- make_vae_dataset.py --environment navreptrain --n 1")
    for i in range(1, 101):
        assert not bash("cp ~/navrep/datasets/V/navreptrain/000_scans_robotstates_actions_rewards_dones.npz \
                        ~/navrep/datasets/V/navreptrain/{:03}_scans_robotstates_actions_rewards_dones.npz".format(
                        i))

@pytest.mark.dependency(depends=["test_navrep_is_clean"])
def test_train_vae1d():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_vae1d.py --environment navreptrain --n 1000")

@pytest.mark.dependency(depends=["test_train_vae1d"])
def test_make_1drnn_dataset():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- make_1drnn_dataset.py --environment navreptrain --n 3")

@pytest.mark.dependency(depends=["test_make_1drnn_dataset"])
def test_train_1drnn():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_rnn.py --environment navreptrain --backend VAE1D_LSTM --n 60")

@pytest.mark.dependency(depends=["test_train_1drnn"])
def test_gym_vae1d_lstm_v_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAE1D_LSTM \
                    --encoding V_ONLY --no-gpu --n 100000")

@pytest.mark.dependency(depends=["test_train_1drnn"])
def test_gym_vae1d_lstm_m_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAE1D_LSTM \
                    --encoding M_ONLY --no-gpu --n 100000")

@pytest.mark.dependency(depends=["test_train_1drnn"])
def test_gym_vae1d_lstm_vm():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAE1D_LSTM \
                    --encoding VM --no-gpu --n 100000")

# VAE_LSTM

@pytest.mark.dependency(depends=["test_navrep_is_clean"])
def test_train_vae():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_vae.py --environment navreptrain --n 1000")

@pytest.mark.dependency(depends=["test_train_vae"])
def test_make_rnn_dataset():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- make_rnn_dataset.py --environment navreptrain --n 3")

@pytest.mark.dependency(depends=["test_make_rnn_dataset"])
def test_train_rnn():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_rnn.py --environment navreptrain --n 60")

@pytest.mark.dependency(depends=["test_train_rnn"])
def test_gym_vae_lstm_v_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAE_LSTM \
                    --encoding V_ONLY --no-gpu --n 100000")

@pytest.mark.dependency(depends=["test_train_rnn"])
def test_gym_vae_lstm_m_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAE_LSTM \
                    --encoding M_ONLY --no-gpu --n 100000")

@pytest.mark.dependency(depends=["test_train_rnn"])
def test_gym_vae_lstm_vm():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAE_LSTM \
                    --encoding VM --no-gpu --n 100000")

# VAE1DLSTM

@pytest.mark.dependency(depends=["test_navrep_is_clean"])
def test_train_vae1dlstm():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_vae1dlstm.py --environment navreptrain --n 100")

@pytest.mark.dependency(depends=["test_train_vaelstm1d"])
def test_gym_vae1dlstm_v_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAE1DLSTM \
                    --encoding V_ONLY --n 100000")

@pytest.mark.dependency(depends=["test_train_vaelstm1d"])
def test_gym_vae1dlstm_m_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAE1DLSTM \
                    --encoding M_ONLY --n 100000")

@pytest.mark.dependency(depends=["test_train_vaelstm1d"])
def test_gym_vae1dlstm_vm():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAE1DLSTM \
                    --encoding VM --n 100000")

# VAELSTM

@pytest.mark.dependency(depends=["test_navrep_is_clean"])
def test_train_vaelstm():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_vaelstm.py --environment navreptrain --n 100")

@pytest.mark.dependency(depends=["test_train_vaelstm"])
def test_gym_vaelstm_v_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAELSTM \
                    --encoding V_ONLY --n 100000")

@pytest.mark.dependency(depends=["test_train_vaelstm"])
def test_gym_vaelstm_m_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAELSTM \
                    --encoding M_ONLY --n 100000")

@pytest.mark.dependency(depends=["test_train_vaelstm"])
def test_gym_vaelstm_vm():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend VAELSTM \
                    --encoding VM --n 100000")

# GPT1D

@pytest.mark.dependency(depends=["test_navrep_is_clean"])
def test_train_gpt1d():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gpt1d.py --environment navreptrain --n 100")

@pytest.mark.dependency(depends=["test_train_gpt1d"])
def test_gym_gpt1d_v_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend GPT1D \
                    --encoding V_ONLY --n 100000")

@pytest.mark.dependency(depends=["test_train_gpt1d"])
def test_gym_gpt1d_m_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend GPT1D \
                    --encoding M_ONLY --n 100000")

@pytest.mark.dependency(depends=["test_train_gpt1d"])
def test_gym_gpt1d_vm():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend GPT1D \
                    --encoding VM --n 100000")

# GPT

@pytest.mark.dependency(depends=["test_navrep_is_clean"])
def test_train_gpt():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gpt.py --environment navreptrain --n 100")

@pytest.mark.dependency(depends=["test_train_gpt"])
def test_gym_gpt_v_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend GPT \
                    --encoding V_ONLY --n 100000")

@pytest.mark.dependency(depends=["test_train_gpt"])
def test_gym_gpt_m_only():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend GPT \
                    --encoding M_ONLY --n 100000")

@pytest.mark.dependency(depends=["test_train_gpt"])
def test_gym_gpt_vm():
    assert not bash("cd ~/Code/navigation_representations && \
                    ipython -- train_gym_navreptrainencodedenv.py --backend GPT \
                    --encoding VM --n 100000")
