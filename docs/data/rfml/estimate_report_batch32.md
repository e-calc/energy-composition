# Kernel-DB energy estimate, level=module (20261003-142845)

E_pred_sum = sum_k e_k = sum_k (e_k - P t_k) + P sum_k t_k ; E_pred_hybrid = sum_k (e_k - P t_k) + P T_e2e

| model | P (W) | hits | keys | T_sum (us) | T_e2e (us) | T err | E_sum (mJ) | E_hyb (mJ) | E_e2e (mJ) | E err sum | E err hyb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rf_vgg | 89.7 | 11/11 | 11 | 670.0 | 672.8 | -0.4% | 100.562 | 100.809 | 107.176 | -6.2% | -5.9% |
| rf_resnet | 89.7 | 10/10 | 10 | 1422.6 | 1488.2 | -4.4% | 216.414 | 222.299 | 223.324 | -3.1% | -0.5% |
| rf_lstm | 89.7 | 3/3 | 3 | 4850.6 | 4814.8 | +0.7% | 898.471 | 895.267 | 909.532 | -1.2% | -1.6% |

- static_power[active_idle] = 89.70 W (n=20, settle=1.0s;std=1.206W;min=86.94;max=90.94)
- static_power[kareus_p2p] = 70.00 W (n=0, constant from kareus tests/bayesian/common/model_config.py GPU_CONFIGS)
- static_power[p8_idle] = 23.39 W (n=3, )
- static_power[post_burst] = 100.41 W (n=20, settle=0.0s;std=17.314W;min=92.24;max=155.71)

## rf_vgg

```
model=rf_vgg level=module gpu=A40 freq=default dtype=fp32 static=active_idle P=89.70 W  kernels hit 11/11 (11 unique keys)
  T_sum   = 670.0 us   T_e2e = 672.8 us   err -0.4%
  E_sum   = 100.562 mJ  (= sum e_k; dyn 40.462 + static 60.100)
  E_hyb   = 100.809 mJ  (= sum E_dyn + P*T_e2e)
  E_e2e   = 107.176 mJ   err sum -6.2%   err hybrid -5.9%

idx type      key                     us        mJ    dyn mJ       W  share  desc / shared
  0 ConvPool  4aa36fd0c98f45f0     120.2    27.754    16.975   231.0  27.6%  ConvPool[2->64,p=448]@32x2x1024
  1 ConvPool  0810d89d591197aa      78.6    18.589    11.543   236.6  18.5%  ConvPool[64->64,p=12352]@32x64x512
  2 ConvPool  7029da0845c2c205      73.7    12.022     5.413   163.2  12.0%  ConvPool[64->64,p=12352]@32x64x256
  3 ConvPool  27631316b07bc57b      74.2     9.164     2.505   123.4   9.1%  ConvPool[64->64,p=12352]@32x64x128
  4 ConvPool  55bf44ca076ddf69      74.6     8.079     1.390   108.3   8.0%  ConvPool[64->64,p=12352]@32x64x64
  5 ConvPool  0e79b1208c6a682d      72.8     7.481     0.952   102.8   7.4%  ConvPool[64->64,p=12352]@32x64x32
  6 ConvPool  1280915c68018e82      73.1     7.426     0.868   101.6   7.4%  ConvPool[64->64,p=12352]@32x64x16
  7 Flatten   ac13ae79a46d53c4       3.5     0.320     0.004    90.9   0.3%  Flatten[p=0]@32x64x8
  8 DenseSELU 16271fb31a81fcf5      37.5     3.689     0.324    98.3   3.7%  DenseSELU[512->128]@32x512  <- rf_resnet
  9 DenseSELU ea2fb69a83feb467      33.7     3.258     0.234    96.6   3.2%  DenseSELU[128->128]@32x128  <- rf_resnet
 10 Classifier 4c32f8878ec0a1be      28.2     2.779     0.254    98.7   2.8%  Classifier[128->24]@32x128  <- rf_resnet
```

## rf_resnet

```
model=rf_resnet level=module gpu=A40 freq=default dtype=fp32 static=active_idle P=89.70 W  kernels hit 10/10 (10 unique keys)
  T_sum   = 1422.6 us   T_e2e = 1488.2 us   err -4.4%
  E_sum   = 216.414 mJ  (= sum e_k; dyn 88.811 + static 127.603)
  E_hyb   = 222.299 mJ  (= sum E_dyn + P*T_e2e)
  E_e2e   = 223.324 mJ   err sum -3.1%   err hybrid -0.5%

idx type      key                     us        mJ    dyn mJ       W  share  desc / shared
  0 ResidualStack 91459f6af71d92c2     234.7    63.376    42.327   270.1  29.3%  ResidualStack[2->32,p=12512]@32x2x1024
  1 ResidualStack 3deebf1706fdc40b     238.5    41.075    19.681   172.2  19.0%  ResidualStack[32->32,p=13472]@32x32x512
  2 ResidualStack dd9bdb767c790991     214.1    30.847    11.640   144.1  14.3%  ResidualStack[32->32,p=13472]@32x32x256
  3 ResidualStack 079dd3e78932dced     213.2    25.493     6.365   119.5  11.8%  ResidualStack[32->32,p=13472]@32x32x128
  4 ResidualStack c729e4006aac41aa     214.3    23.766     4.545   110.9  11.0%  ResidualStack[32->32,p=13472]@32x32x64
  5 ResidualStack 7393e8215d10caea     204.8    21.804     3.431   106.4  10.1%  ResidualStack[32->32,p=13472]@32x32x32
  6 Flatten   a5f08415d4469cc8       3.5     0.326     0.010    92.6   0.2%  Flatten[p=0]@32x32x16
  7 DenseSELU 16271fb31a81fcf5      37.5     3.689     0.324    98.3   1.7%  DenseSELU[512->128]@32x512  <- rf_vgg
  8 DenseSELU ea2fb69a83feb467      33.7     3.258     0.234    96.6   1.5%  DenseSELU[128->128]@32x128  <- rf_vgg
  9 Classifier 4c32f8878ec0a1be      28.2     2.779     0.254    98.7   1.3%  Classifier[128->24]@32x128  <- rf_vgg
```

## rf_lstm

```
model=rf_lstm level=module gpu=A40 freq=default dtype=fp32 static=active_idle P=89.70 W  kernels hit 3/3 (3 unique keys)
  T_sum   = 4850.6 us   T_e2e = 4814.8 us   err +0.7%
  E_sum   = 898.471 mJ  (= sum e_k; dyn 463.385 + static 435.086)
  E_hyb   = 895.267 mJ  (= sum E_dyn + P*T_e2e)
  E_e2e   = 909.532 mJ   err sum -1.2%   err hybrid -1.6%

idx type      key                     us        mJ    dyn mJ       W  share  desc / shared
  0 LSTMLayer 3126292238d21487    2368.4   429.542   217.100   181.4  47.8%  LSTMLayer[2->128]@32x1024x2
  1 LSTMLayer 01464e3d1a62b133    2449.7   465.657   245.922   190.1  51.8%  LSTMLayer[128->128]@32x1024x128
  2 LastStepClassifier 2e6cff0015d4ab12      32.4     3.271     0.363   100.9   0.4%  LastStepClassifier[128->24]@32x1024x128
```

## Kernel reuse (occurrences per model)

| key | desc | rf_vgg | rf_resnet | rf_lstm |
|---|---|---|---|---|
| 4c32f8878ec0a1be | Classifier[128->24]@32x128 | 1 | 1 | 0 |
| 4aa36fd0c98f45f0 | ConvPool[2->64,p=448]@32x2x1024 | 1 | 0 | 0 |
| 27631316b07bc57b | ConvPool[64->64,p=12352]@32x64x128 | 1 | 0 | 0 |
| 1280915c68018e82 | ConvPool[64->64,p=12352]@32x64x16 | 1 | 0 | 0 |
| 7029da0845c2c205 | ConvPool[64->64,p=12352]@32x64x256 | 1 | 0 | 0 |
| 0e79b1208c6a682d | ConvPool[64->64,p=12352]@32x64x32 | 1 | 0 | 0 |
| 0810d89d591197aa | ConvPool[64->64,p=12352]@32x64x512 | 1 | 0 | 0 |
| 55bf44ca076ddf69 | ConvPool[64->64,p=12352]@32x64x64 | 1 | 0 | 0 |
| ea2fb69a83feb467 | DenseSELU[128->128]@32x128 | 1 | 1 | 0 |
| 16271fb31a81fcf5 | DenseSELU[512->128]@32x512 | 1 | 1 | 0 |
| a5f08415d4469cc8 | Flatten[p=0]@32x32x16 | 0 | 1 | 0 |
| ac13ae79a46d53c4 | Flatten[p=0]@32x64x8 | 1 | 0 | 0 |
| 01464e3d1a62b133 | LSTMLayer[128->128]@32x1024x128 | 0 | 0 | 1 |
| 3126292238d21487 | LSTMLayer[2->128]@32x1024x2 | 0 | 0 | 1 |
| 2e6cff0015d4ab12 | LastStepClassifier[128->24]@32x1024x128 | 0 | 0 | 1 |
| 91459f6af71d92c2 | ResidualStack[2->32,p=12512]@32x2x1024 | 0 | 1 | 0 |
| 079dd3e78932dced | ResidualStack[32->32,p=13472]@32x32x128 | 0 | 1 | 0 |
| dd9bdb767c790991 | ResidualStack[32->32,p=13472]@32x32x256 | 0 | 1 | 0 |
| 7393e8215d10caea | ResidualStack[32->32,p=13472]@32x32x32 | 0 | 1 | 0 |
| 3deebf1706fdc40b | ResidualStack[32->32,p=13472]@32x32x512 | 0 | 1 | 0 |
| c729e4006aac41aa | ResidualStack[32->32,p=13472]@32x32x64 | 0 | 1 | 0 |
