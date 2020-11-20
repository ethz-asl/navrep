set -e  # exit if any command fails
set -x  # print commands being executed
# VAE1D_LSTM
python -m navrep.scripts.make_vae_dataset --environment navreptrain
python -m navrep.scripts.train_vae1d --environment navreptrain
python -m navrep.scripts.make_1drnn_dataset --environment navreptrain
python -m navrep.scripts.train_rnn --environment navreptrain --backend VAE1D_LSTM
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAE1D_LSTM --encoding V_ONLY --no-gpu --n 100000000
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAE1D_LSTM --encoding M_ONLY --no-gpu --n 100000000
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAE1D_LSTM --encoding VM --no-gpu --n 100000000
# VAE_LSTM
python -m navrep.scripts.make_vae_dataset --environment navreptrain
python -m navrep.scripts.train_vae --environment navreptrain
python -m navrep.scripts.make_rnn_dataset --environment navreptrain
python -m navrep.scripts.train_rnn --environment navreptrain
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAE_LSTM --encoding V_ONLY --no-gpu --n 100000000
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAE_LSTM --encoding M_ONLY --no-gpu --n 100000000
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAE_LSTM --encoding VM --no-gpu --n 100000000
# VAELSTM
python -m navrep.scripts.make_vae_dataset --environment navreptrain
python -m navrep.scripts.train_vaelstm --environment navreptrain
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAELSTM --encoding V_ONLY
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAELSTM --encoding M_ONLY
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAELSTM --encoding VM
# VAE1DLSTM
python -m navrep.scripts.make_vae_dataset --environment navreptrain
python -m navrep.scripts.train_vae1dlstm --environment navreptrain
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAE1DLSTM --encoding V_ONLY
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAE1DLSTM --encoding M_ONLY
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend VAE1DLSTM --encoding VM
# GPT
python -m navrep.scripts.make_vae_dataset --environment navreptrain
python -m navrep.scripts.train_gpt --environment navreptrain
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend GPT --encoding V_ONLY
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend GPT --encoding M_ONLY
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend GPT --encoding VM
# GPT 1D
python -m navrep.scripts.make_vae_dataset --environment navreptrain
python -m navrep.scripts.train_gpt1d --environment navreptrain
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend GPT1D --encoding V_ONLY
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend GPT1D --encoding M_ONLY
python -m navrep.scripts.train_gym_navreptrainencodedenv --backend GPT1D --encoding VM
# E2E
python -m navrep.scripts.train_gym_e2enavreptrainenv
# E2E1D
python -m navrep.scripts.train_gym_e2e1dnavreptrainenv

# cross test
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAE1D_LSTM --encoding V_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAE1D_LSTM --encoding M_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAE1D_LSTM --encoding VM
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAE_LSTM --encoding V_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAE_LSTM --encoding M_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAE_LSTM --encoding VM
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAE1DLSTM --encoding V_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAE1DLSTM --encoding M_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAE1DLSTM --encoding VM
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAELSTM --encoding V_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAELSTM --encoding M_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend VAELSTM --encoding VM
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend GPT1D --encoding V_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend GPT1D --encoding M_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend GPT1D --encoding VM
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend GPT --encoding V_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend GPT --encoding M_ONLY
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend GPT --encoding VM
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend E2E --encoding VCARCH
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --no-gpu --backend E2E1D --encoding VCARCH
