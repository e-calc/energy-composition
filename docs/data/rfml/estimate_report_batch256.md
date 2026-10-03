# Kernel-DB energy estimate, level=module (20261003-192617)

E_pred_sum = sum_k e_k = sum_k (e_k - P t_k) + P sum_k t_k ; E_pred_hybrid = sum_k (e_k - P t_k) + P T_e2e

| model | P (W) | hits | keys | T_sum (us) | T_e2e (us) | T err | E_sum (mJ) | E_hyb (mJ) | E_e2e (mJ) | E err sum | E err hyb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rf_vgg | 89.7 | 11/11 | 11 | 2434.1 | 2280.0 | +6.8% | 631.397 | 617.577 | 615.709 | +2.5% | +0.3% |
| rf_resnet | 89.7 | 10/10 | 10 | 4590.1 | 4250.7 | +8.0% | 1258.694 | 1228.254 | 1240.046 | +1.5% | -1.0% |
| rf_lstm | 89.7 | 3/3 | 3 | 17363.0 | 17368.6 | -0.0% | 5032.349 | 5032.851 | 5024.406 | +0.2% | +0.2% |

- static_power[active_idle] = 89.70 W (n=20, settle=1.0s;std=1.206W;min=86.94;max=90.94)
- static_power[kareus_p2p] = 70.00 W (n=0, constant from kareus tests/bayesian/common/model_config.py GPU_CONFIGS)
- static_power[p8_idle] = 23.39 W (n=3, )
- static_power[post_burst] = 100.41 W (n=20, settle=0.0s;std=17.314W;min=92.24;max=155.71)

## rf_vgg

```
model=rf_vgg level=module gpu=A40 freq=default dtype=fp32 static=active_idle P=89.70 W  kernels hit 11/11 (11 unique keys)
  T_sum   = 2434.1 us   T_e2e = 2280.0 us   err +6.8%
  E_sum   = 631.397 mJ  (= sum e_k; dyn 413.067 + static 218.330)
  E_hyb   = 617.577 mJ  (= sum E_dyn + P*T_e2e)
  E_e2e   = 615.709 mJ   err sum +2.5%   err hybrid +0.3%

idx type      key                     us        mJ    dyn mJ       W  share  desc / shared
  0 ConvPool  1fe06ba3d8b42abc     925.9   248.286   165.235   268.2  39.3%  ConvPool[2->64,p=448]@256x2x1024
  1 ConvPool  088468d743c31d56     658.6   188.600   129.528   286.4  29.9%  ConvPool[64->64,p=12352]@256x64x512
  2 ConvPool  d50af392910f52dc     340.3    94.754    64.226   278.4  15.0%  ConvPool[64->64,p=12352]@256x64x256
  3 ConvPool  b3a023cc4873ba9d     184.5    48.252    31.702   261.5   7.6%  ConvPool[64->64,p=12352]@256x64x128
  4 ConvPool  80d318443f60409a      78.6    19.628    12.575   249.6   3.1%  ConvPool[64->64,p=12352]@256x64x64
  5 ConvPool  d407c38351df1fc7      71.2    11.955     5.565   167.8   1.9%  ConvPool[64->64,p=12352]@256x64x32
  6 ConvPool  bd5780891a3ccaa2      71.0     9.234     2.869   130.1   1.5%  ConvPool[64->64,p=12352]@256x64x16
  7 Flatten   46313e35239090f1       3.5     0.322     0.011    93.0   0.1%  Flatten[p=0]@256x64x8
  8 DenseSELU a4048f70448fdfcc      37.5     4.165     0.798   111.0   0.7%  DenseSELU[512->128]@256x512  <- rf_resnet
  9 DenseSELU 0c4ed5fdf55bd1ab      34.4     3.363     0.278    97.8   0.5%  DenseSELU[128->128]@256x128  <- rf_resnet
 10 Classifier b82ca0bc94382ff6      28.5     2.839     0.280    99.5   0.4%  Classifier[128->24]@256x128  <- rf_resnet
```

## rf_resnet

```
model=rf_resnet level=module gpu=A40 freq=default dtype=fp32 static=active_idle P=89.70 W  kernels hit 10/10 (10 unique keys)
  T_sum   = 4590.1 us   T_e2e = 4250.7 us   err +8.0%
  E_sum   = 1258.694 mJ  (= sum e_k; dyn 846.974 + static 411.720)
  E_hyb   = 1228.254 mJ  (= sum E_dyn + P*T_e2e)
  E_e2e   = 1240.046 mJ   err sum +1.5%   err hybrid -1.0%

idx type      key                     us        mJ    dyn mJ       W  share  desc / shared
  0 ResidualStack 8f8af84f49daf4ef    2086.2   623.174   436.048   298.7  49.5%  ResidualStack[2->32,p=12512]@256x2x1024
  1 ResidualStack 85bac86935fecd37    1097.7   324.721   226.258   295.8  25.8%  ResidualStack[32->32,p=13472]@256x32x512
  2 ResidualStack 091eaa596da13470     580.6   160.313   108.237   276.1  12.7%  ResidualStack[32->32,p=13472]@256x32x256
  3 ResidualStack 53de420859514dd7     248.3    66.032    43.756   265.9   5.2%  ResidualStack[32->32,p=13472]@256x32x128
  4 ResidualStack 593f480bb747b974     248.3    42.031    19.758   169.3   3.3%  ResidualStack[32->32,p=13472]@256x32x64
  5 ResidualStack f13292abaf4c5bc9     225.0    31.728    11.545   141.0   2.5%  ResidualStack[32->32,p=13472]@256x32x32
  6 Flatten   c4907177d0969c5c       3.5     0.328     0.014    93.8   0.0%  Flatten[p=0]@256x32x16
  7 DenseSELU a4048f70448fdfcc      37.5     4.165     0.798   111.0   0.3%  DenseSELU[512->128]@256x512  <- rf_vgg
  8 DenseSELU 0c4ed5fdf55bd1ab      34.4     3.363     0.278    97.8   0.3%  DenseSELU[128->128]@256x128  <- rf_vgg
  9 Classifier b82ca0bc94382ff6      28.5     2.839     0.280    99.5   0.2%  Classifier[128->24]@256x128  <- rf_vgg
```

## rf_lstm

```
model=rf_lstm level=module gpu=A40 freq=default dtype=fp32 static=active_idle P=89.70 W  kernels hit 3/3 (3 unique keys)
  T_sum   = 17363.0 us   T_e2e = 17368.6 us   err -0.0%
  E_sum   = 5032.349 mJ  (= sum e_k; dyn 3474.922 + static 1557.427)
  E_hyb   = 5032.851 mJ  (= sum E_dyn + P*T_e2e)
  E_e2e   = 5024.406 mJ   err sum +0.2%   err hybrid +0.2%

idx type      key                     us        mJ    dyn mJ       W  share  desc / shared
  0 LSTMLayer bf049206d7c9a722    8521.0  2447.868  1683.547   287.3  48.6%  LSTMLayer[2->128]@256x1024x2
  1 LSTMLayer 80f8bf70347d00b1    8808.5  2581.265  1791.156   293.0  51.3%  LSTMLayer[128->128]@256x1024x128
  2 LastStepClassifier 7ee554b7b7de5588      33.4     3.217     0.219    96.3   0.1%  LastStepClassifier[128->24]@256x1024x128
```

## Kernel reuse (occurrences per model)

| key | desc | rf_vgg | rf_resnet | rf_lstm |
|---|---|---|---|---|
| b82ca0bc94382ff6 | Classifier[128->24]@256x128 | 1 | 1 | 0 |
| 1fe06ba3d8b42abc | ConvPool[2->64,p=448]@256x2x1024 | 1 | 0 | 0 |
| b3a023cc4873ba9d | ConvPool[64->64,p=12352]@256x64x128 | 1 | 0 | 0 |
| bd5780891a3ccaa2 | ConvPool[64->64,p=12352]@256x64x16 | 1 | 0 | 0 |
| d50af392910f52dc | ConvPool[64->64,p=12352]@256x64x256 | 1 | 0 | 0 |
| d407c38351df1fc7 | ConvPool[64->64,p=12352]@256x64x32 | 1 | 0 | 0 |
| 088468d743c31d56 | ConvPool[64->64,p=12352]@256x64x512 | 1 | 0 | 0 |
| 80d318443f60409a | ConvPool[64->64,p=12352]@256x64x64 | 1 | 0 | 0 |
| 0c4ed5fdf55bd1ab | DenseSELU[128->128]@256x128 | 1 | 1 | 0 |
| a4048f70448fdfcc | DenseSELU[512->128]@256x512 | 1 | 1 | 0 |
| c4907177d0969c5c | Flatten[p=0]@256x32x16 | 0 | 1 | 0 |
| 46313e35239090f1 | Flatten[p=0]@256x64x8 | 1 | 0 | 0 |
| 80f8bf70347d00b1 | LSTMLayer[128->128]@256x1024x128 | 0 | 0 | 1 |
| bf049206d7c9a722 | LSTMLayer[2->128]@256x1024x2 | 0 | 0 | 1 |
| 7ee554b7b7de5588 | LastStepClassifier[128->24]@256x1024x128 | 0 | 0 | 1 |
| 8f8af84f49daf4ef | ResidualStack[2->32,p=12512]@256x2x1024 | 0 | 1 | 0 |
| 53de420859514dd7 | ResidualStack[32->32,p=13472]@256x32x128 | 0 | 1 | 0 |
| 091eaa596da13470 | ResidualStack[32->32,p=13472]@256x32x256 | 0 | 1 | 0 |
| f13292abaf4c5bc9 | ResidualStack[32->32,p=13472]@256x32x32 | 0 | 1 | 0 |
| 85bac86935fecd37 | ResidualStack[32->32,p=13472]@256x32x512 | 0 | 1 | 0 |
| 593f480bb747b974 | ResidualStack[32->32,p=13472]@256x32x64 | 0 | 1 | 0 |
