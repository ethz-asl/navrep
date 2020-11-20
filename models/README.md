## Pre-trained Models

This folder contains the pre-trained models which were evaluated in the NavRep paper.

### V, M, W

The V, M, and W folders contain the trained unsupervised models which perform feature extraction from lidar data.

Modular variants have the V and M modules saved separately (inside the V and M folders),
Joint variants are saved inside the W folders, as a single model.

```
models
├── M
│   ├── navreptrainrnn1d.json  (Modular variant with 1d lidar representation, V module)
│   └── navreptrainrnn.json    (Modular variant with rings lidar representation, V module)
├── V
│   ├── navreptrainvae1d.json  (Modular variant with 1d lidar representation, M module)
│   └── navreptrainvae.json    (Modular variant with rings lidar representation, M module)
└── W
    ├── navreptraingpt         (Transformer variant with rings lidar representation)
    ├── navreptraingpt1d       (Transformer variant with 1d lidar representation)
    ├── navreptrainvae1dlstm   (Joint variant with 1d lidar representation)
    └── navreptrainvaelstm     (Joint variant with rings lidar representation)
````

### C models

The gym folder contains the trained C models. A breakdown of the model file names:

```navreptrainencodedenv_2020_09_17__09_15_17_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip```

- `navreptrainencodedenv`: the environment in which the C model was trained
- `2020_09_17__09_15_17`: the date at which the C model was trained
- `PPO`: the RL algorithm used to train the C model
- `VAE_LSTM`: the architecture variant (`VAE1D_LSTM`, `VAE_LSTM`, `VAE1DLSTM`, `VAELSTM`, `GPT1D`, `GPT`, `E2E`, `E2E1D`)
- `V_ONLY`: the input features variant (`V_ONLY`, `M_ONLY`, `VM`)
- `V32M512`: the input features dimensions

### Example

For example, to test a *Modular-architecture, 1d lidar representation, z-features-only* C model,
one needs the C model itself and the relevant V and M modules, like so:
```
~/navrep/
  └── models
      ├── gym
      │   └── navreptrainencodedenv_2020_09_17__09_15_17_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip
      ├── M
      │   └── navreptrainrnn.json
      └── V
          └── navreptrainvae.json
```

```
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --backend VAE_LSTM --encoding V_ONLY --render
```


To test a *Transformer-architecture (joint), rings lidar representation, z+h-features* C model:

```
~/navrep/
  └── models
      ├── gym
      │   └── navreptrainencodedenv_2020_10_02__23_50_56_PPO_GPT_VM_V64M64_ckpt.zip
      └── W
          └── navreptraingpt
```

```
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --backend GPT --encoding VM --render
```

To test an *End-to-end architecture, rings lidar representation* C model:
```
~/navrep/
  └── models
      └── gym
          └── e2enavreptrainenv_2020_09_08__14_01_44_PPO_E2E_VCARCH_C64_ckpt.zip
```

```
python -m navrep.scripts.cross_test_navreptrain_in_ianenv --backend E2E --encoding VCARCH --render
```
