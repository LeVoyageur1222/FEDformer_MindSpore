### Experiments Results

#### Original Results

![img](https://user-images.githubusercontent.com/44238026/171345192-e7440898-4019-4051-86e0-681d1a28d630.png)

#### Reproduction Results

![image-20250615164731885](C:\Users\XHZ\AppData\Roaming\Typora\typora-user-images\image-20250615164731885.png)

### Experiments Logs

#### Log for Transformer

```bash
Args in experiment:
Namespace(is_training=1, task_id='Exchange', model='Transformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/exchange_rate/', data_path='exchange_rate.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=8, dec_in=8, c_out=8, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
>>>>>>> start training : Exchange_Transformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 160/160 [00:27<00:00,  5.76it/s]
Epoch 1: Train Loss 0.882124 | Val Loss 1.827491 | Test Loss 3.215585
Validation loss decreased (inf --> 1.827491). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 160/160 [00:26<00:00,  6.05it/s]
Epoch 2: Train Loss 0.738337 | Val Loss 2.516119 | Test Loss 4.239336
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 160/160 [00:26<00:00,  6.15it/s]
Epoch 3: Train Loss 0.749231 | Val Loss 3.625508 | Test Loss 5.308573
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
100%|█████████████████████████████████████████| 160/160 [00:26<00:00,  6.09it/s]
Epoch 4: Train Loss 1.206812 | Val Loss 5.042057 | Test Loss 3.494694
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>> testing : Exchange_Transformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 3.183134078979492, MAE: 1.4182780981063843, RMSE: 1.784134030342102, MAPE: 12.24740982055664, MSPE: 49702.41796875
Args in experiment:
Namespace(is_training=1, task_id='weather', model='Transformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/weather/', data_path='weather.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=21, dec_in=21, c_out=21, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
>>>>>>> start training : weather_Transformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|███████████████████████████████████████| 1146/1146 [03:11<00:00,  5.97it/s]
Epoch 1: Train Loss 0.731208 | Val Loss 2.834105 | Test Loss 4.272011
Validation loss decreased (inf --> 2.834105). Saving model ...
Updating learning rate to 0.0001
100%|███████████████████████████████████████| 1146/1146 [03:09<00:00,  6.04it/s]
Epoch 2: Train Loss 0.733534 | Val Loss 3.065521 | Test Loss 4.740158
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
100%|███████████████████████████████████████| 1146/1146 [03:09<00:00,  6.04it/s]
Epoch 3: Train Loss 0.749875 | Val Loss 4.124500 | Test Loss 5.982051
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
100%|███████████████████████████████████████| 1146/1146 [03:09<00:00,  6.05it/s]
Epoch 4: Train Loss 0.791824 | Val Loss 5.283945 | Test Loss 7.453146
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>> testing : weather_Transformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 4.270166397094727, MAE: 1.5801911354064941, RMSE: 2.0664381980895996, MAPE: 25.459999084472656, MSPE: 31334090.0
Args in experiment:
Namespace(is_training=1, task_id='ECL', model='Transformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/electricity/', data_path='electricity.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=321, dec_in=321, c_out=321, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
>>>>>>> start training : ECL_Transformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 569/569 [02:35<00:00,  3.66it/s]
Epoch 1: Train Loss 0.929257 | Val Loss 2.786046 | Test Loss 3.538372
Validation loss decreased (inf --> 2.786046). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 569/569 [02:32<00:00,  3.72it/s]
Epoch 2: Train Loss 0.931936 | Val Loss 3.021109 | Test Loss 3.631670
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 569/569 [02:34<00:00,  3.68it/s]
Epoch 3: Train Loss 0.952029 | Val Loss 2.813809 | Test Loss 3.396411
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
100%|█████████████████████████████████████████| 569/569 [02:33<00:00,  3.70it/s]
Epoch 4: Train Loss 0.960459 | Val Loss 1.748822 | Test Loss 2.410612
Validation loss decreased (2.786046 --> 1.748822). Saving model ...
Updating learning rate to 1.25e-05
100%|█████████████████████████████████████████| 569/569 [02:33<00:00,  3.69it/s]
Epoch 5: Train Loss 0.994286 | Val Loss 1.166870 | Test Loss 1.567483
Validation loss decreased (1.748822 --> 1.166870). Saving model ...
Updating learning rate to 6.25e-06
100%|█████████████████████████████████████████| 569/569 [02:33<00:00,  3.70it/s]
Epoch 6: Train Loss 0.993456 | Val Loss 1.018694 | Test Loss 1.201009
Validation loss decreased (1.166870 --> 1.018694). Saving model ...
Updating learning rate to 3.125e-06
100%|█████████████████████████████████████████| 569/569 [02:34<00:00,  3.68it/s]
Epoch 7: Train Loss 1.006230 | Val Loss 0.933280 | Test Loss 1.100392
Validation loss decreased (1.018694 --> 0.933280). Saving model ...
Updating learning rate to 1.5625e-06
100%|█████████████████████████████████████████| 569/569 [02:34<00:00,  3.69it/s]
Epoch 8: Train Loss 1.001355 | Val Loss 0.908273 | Test Loss 1.063240
Validation loss decreased (0.933280 --> 0.908273). Saving model ...
Updating learning rate to 7.8125e-07
100%|█████████████████████████████████████████| 569/569 [02:33<00:00,  3.70it/s]
Epoch 9: Train Loss 0.998098 | Val Loss 0.897207 | Test Loss 1.038031
Validation loss decreased (0.908273 --> 0.897207). Saving model ...
Updating learning rate to 3.90625e-07
100%|█████████████████████████████████████████| 569/569 [02:33<00:00,  3.70it/s]
Epoch 10: Train Loss 0.995703 | Val Loss 0.891777 | Test Loss 1.028797
Validation loss decreased (0.897207 --> 0.891777). Saving model ...
Updating learning rate to 1.953125e-07
>>>>>>> testing : ECL_Transformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 1.0280888080596924, MAE: 0.8295090794563293, RMSE: 1.0139471292495728, MAPE: 3.2383921146392822, MSPE: 628375.25
Args in experiment:
Namespace(is_training=1, task_id='traffic', model='Transformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/traffic/', data_path='traffic.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=862, dec_in=862, c_out=862, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=3, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
>>>>>>> start training : traffic_Transformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 377/377 [02:57<00:00,  2.13it/s]
Epoch 1: Train Loss 1.009887 | Val Loss 1.546621 | Test Loss 1.802800
Validation loss decreased (inf --> 1.546621). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 377/377 [02:54<00:00,  2.17it/s]
Epoch 2: Train Loss 0.992764 | Val Loss 1.580348 | Test Loss 1.840194
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 377/377 [02:55<00:00,  2.15it/s]
Epoch 3: Train Loss 0.995501 | Val Loss 1.482804 | Test Loss 1.726366
Validation loss decreased (1.546621 --> 1.482804). Saving model ...
Updating learning rate to 2.5e-05
>>>>>>> testing : traffic_Transformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 1.7281954288482666, MAE: 0.9112874269485474, RMSE: 1.3146084547042847, MAPE: 5.847187519073486, MSPE: 321727.28125
```

#### Log for Informer

```bash
Args in experiment:
Namespace(is_training=1, task_id='Exchange', model='Informer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/exchange_rate/', data_path='exchange_rate.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=8, dec_in=8, c_out=8, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
>>>>>>> start training : Exchange_Informer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 160/160 [06:51<00:00,  2.57s/it]
Epoch 1: Train Loss 0.935397 | Val Loss 3.177367 | Test Loss 3.836076
Validation loss decreased (inf --> 3.177367). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 160/160 [06:51<00:00,  2.57s/it]
Epoch 2: Train Loss 0.956693 | Val Loss 3.950173 | Test Loss 3.852087
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 160/160 [06:51<00:00,  2.57s/it]
Epoch 3: Train Loss 1.006362 | Val Loss 5.982340 | Test Loss 3.677508
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
100%|█████████████████████████████████████████| 160/160 [06:49<00:00,  2.56s/it]
Epoch 4: Train Loss 1.022042 | Val Loss 5.876305 | Test Loss 3.449150
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>> testing : Exchange_Informer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 3.8173670768737793, MAE: 1.5888177156448364, RMSE: 1.953808307647705, MAPE: 9.692940711975098, MSPE: 31511.759765625
Args in experiment:
Namespace(is_training=1, task_id='weather', model='Informer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/weather/', data_path='weather.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=21, dec_in=21, c_out=21, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
>>>>>>> start training : weather_Informer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|███████████████████████████████████████| 1146/1146 [47:37<00:00,  2.49s/it]
Args in experiment:
Namespace(is_training=1, task_id='ECL', model='Informer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/electricity/', data_path='electricity.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=321, dec_in=321, c_out=321, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
>>>>>>> start training : ECL_Informer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 569/569 [23:55<00:00,  2.52s/it]
Epoch 1: Train Loss 0.927152 | Val Loss 3.101039 | Test Loss 3.847202
Validation loss decreased (inf --> 3.101039). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 569/569 [23:24<00:00,  2.47s/it]
Epoch 2: Train Loss 0.915685 | Val Loss 2.958141 | Test Loss 3.673228
Validation loss decreased (3.101039 --> 2.958141). Saving model ...
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 569/569 [23:22<00:00,  2.47s/it]
Epoch 3: Train Loss 0.934864 | Val Loss 1.983242 | Test Loss 2.646787
Validation loss decreased (2.958141 --> 1.983242). Saving model ...
Updating learning rate to 2.5e-05
100%|█████████████████████████████████████████| 569/569 [23:12<00:00,  2.45s/it]
Epoch 4: Train Loss 0.976062 | Val Loss 1.534732 | Test Loss 1.708982
Validation loss decreased (1.983242 --> 1.534732). Saving model ...
Updating learning rate to 1.25e-05
100%|█████████████████████████████████████████| 569/569 [23:16<00:00,  2.45s/it]
Epoch 5: Train Loss 0.944821 | Val Loss 1.380558 | Test Loss 1.528210
Validation loss decreased (1.534732 --> 1.380558). Saving model ...
Updating learning rate to 6.25e-06
100%|█████████████████████████████████████████| 569/569 [23:21<00:00,  2.46s/it]
Epoch 6: Train Loss 0.930896 | Val Loss 1.286173 | Test Loss 1.408471
Validation loss decreased (1.380558 --> 1.286173). Saving model ...
Updating learning rate to 3.125e-06
100%|█████████████████████████████████████████| 569/569 [23:12<00:00,  2.45s/it]
Epoch 7: Train Loss 0.931531 | Val Loss 1.206738 | Test Loss 1.303152
Validation loss decreased (1.286173 --> 1.206738). Saving model ...
Updating learning rate to 1.5625e-06
100%|█████████████████████████████████████████| 569/569 [23:18<00:00,  2.46s/it]
Epoch 8: Train Loss 0.934148 | Val Loss 1.204219 | Test Loss 1.291163
Validation loss decreased (1.206738 --> 1.204219). Saving model ...
Updating learning rate to 7.8125e-07
100%|█████████████████████████████████████████| 569/569 [23:21<00:00,  2.46s/it]
Epoch 9: Train Loss 0.929780 | Val Loss 1.202739 | Test Loss 1.286663
Validation loss decreased (1.204219 --> 1.202739). Saving model ...
Updating learning rate to 3.90625e-07
100%|█████████████████████████████████████████| 569/569 [23:28<00:00,  2.47s/it]
Epoch 10: Train Loss 0.928013 | Val Loss 1.201770 | Test Loss 1.284255
Validation loss decreased (1.202739 --> 1.201770). Saving model ...
Updating learning rate to 1.953125e-07
>>>>>>> testing : ECL_Informer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 1.2835747003555298, MAE: 0.9381105899810791, RMSE: 1.1329495906829834, MAPE: 5.319793224334717, MSPE: 193859.453125
Args in experiment:
Namespace(is_training=1, task_id='traffic', model='Informer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/traffic/', data_path='traffic.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=862, dec_in=862, c_out=862, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=3, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
>>>>>>> start training : traffic_Informer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 377/377 [17:54<00:00,  2.85s/it]
Epoch 1: Train Loss 1.006298 | Val Loss 1.564734 | Test Loss 1.826361
Validation loss decreased (inf --> 1.564734). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 377/377 [17:45<00:00,  2.82s/it]
Epoch 2: Train Loss 0.991120 | Val Loss 1.636981 | Test Loss 1.921199
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 377/377 [16:49<00:00,  2.68s/it]
Epoch 3: Train Loss 0.988256 | Val Loss 1.631184 | Test Loss 1.913476
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
>>>>>>> testing : traffic_Informer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 1.8315457105636597, MAE: 0.9633457660675049, RMSE: 1.3533461093902588, MAPE: 7.461945533752441, MSPE: 883151.3125
```

#### Log for Autoformer

```bash
Args in experiment:
Namespace(is_training=1, task_id='Exchange', model='Autoformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/exchange_rate/', data_path='exchange_rate.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=8, dec_in=8, c_out=8, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
AutoCorrelation used!
AutoCorrelation used!
AutoCorrelation used!
AutoCorrelation used!
>>>>>>> start training : Exchange_Autoformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 160/160 [03:37<00:00,  1.36s/it]
Epoch 1: Train Loss 0.231729 | Val Loss 0.185220 | Test Loss 0.149846
Validation loss decreased (inf --> 0.185220). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 160/160 [03:30<00:00,  1.31s/it]
Epoch 2: Train Loss 0.214482 | Val Loss 0.184336 | Test Loss 0.148587
Validation loss decreased (0.185220 --> 0.184336). Saving model ...
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 160/160 [03:32<00:00,  1.33s/it]
Epoch 3: Train Loss 0.211816 | Val Loss 0.185403 | Test Loss 0.146147
EarlyStopping counter: 1 out of 3
Updating learning rate to 2.5e-05
100%|█████████████████████████████████████████| 160/160 [03:30<00:00,  1.31s/it]
Epoch 4: Train Loss 0.210309 | Val Loss 0.185262 | Test Loss 0.146231
EarlyStopping counter: 2 out of 3
Updating learning rate to 1.25e-05
100%|█████████████████████████████████████████| 160/160 [03:30<00:00,  1.32s/it]
Epoch 5: Train Loss 0.209770 | Val Loss 0.185171 | Test Loss 0.146352
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>> testing : Exchange_Autoformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 0.14855751395225525, MAE: 0.2783169746398926, RMSE: 0.38543158769607544, MAPE: 1.6906903982162476, MSPE: 1645.2667236328125
Args in experiment:
Namespace(is_training=1, task_id='weather', model='Autoformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/weather/', data_path='weather.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=21, dec_in=21, c_out=21, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
AutoCorrelation used!
AutoCorrelation used!
AutoCorrelation used!
AutoCorrelation used!
>>>>>>> start training : weather_Autoformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|███████████████████████████████████████| 1146/1146 [24:46<00:00,  1.30s/it]
Epoch 1: Train Loss 0.662518 | Val Loss 16.653643 | Test Loss 14.509151
Validation loss decreased (inf --> 16.653643). Saving model ...
Updating learning rate to 0.0001
100%|███████████████████████████████████████| 1146/1146 [23:54<00:00,  1.25s/it]
Epoch 2: Train Loss 0.626683 | Val Loss 29.055374 | Test Loss 23.966343
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
100%|███████████████████████████████████████| 1146/1146 [24:09<00:00,  1.26s/it]
Epoch 3: Train Loss 0.619704 | Val Loss 31.091183 | Test Loss 24.149635
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
100%|███████████████████████████████████████| 1146/1146 [24:14<00:00,  1.27s/it]
Epoch 4: Train Loss 0.613803 | Val Loss 28.421347 | Test Loss 22.740540
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>> testing : weather_Autoformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 14.414979934692383, MAE: 2.378880023956299, RMSE: 3.796706438064575, MAPE: 87.03179168701172, MSPE: 730062848.0
Args in experiment:
Namespace(is_training=1, task_id='ECL', model='Autoformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/electricity/', data_path='electricity.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=321, dec_in=321, c_out=321, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
AutoCorrelation used!
AutoCorrelation used!
AutoCorrelation used!
AutoCorrelation used!
>>>>>>> start training : ECL_Autoformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 569/569 [13:20<00:00,  1.41s/it]
Epoch 1: Train Loss 0.409331 | Val Loss 4.746860 | Test Loss 4.802910
Validation loss decreased (inf --> 4.746860). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 569/569 [13:22<00:00,  1.41s/it]
Epoch 2: Train Loss 0.316689 | Val Loss 4.435345 | Test Loss 4.440558
Validation loss decreased (4.746860 --> 4.435345). Saving model ...
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 569/569 [13:29<00:00,  1.42s/it]
Epoch 3: Train Loss 0.310380 | Val Loss 4.336590 | Test Loss 4.336375
Validation loss decreased (4.435345 --> 4.336590). Saving model ...
Updating learning rate to 2.5e-05
100%|█████████████████████████████████████████| 569/569 [13:20<00:00,  1.41s/it]
Epoch 4: Train Loss 0.311213 | Val Loss 4.481692 | Test Loss 4.452764
EarlyStopping counter: 1 out of 3
Updating learning rate to 1.25e-05
100%|█████████████████████████████████████████| 569/569 [13:03<00:00,  1.38s/it]
Epoch 5: Train Loss 0.312506 | Val Loss 4.237007 | Test Loss 4.275901
Validation loss decreased (4.336590 --> 4.237007). Saving model ...
Updating learning rate to 6.25e-06
100%|█████████████████████████████████████████| 569/569 [13:03<00:00,  1.38s/it]
Epoch 6: Train Loss 0.312989 | Val Loss 4.128956 | Test Loss 4.051245
Validation loss decreased (4.237007 --> 4.128956). Saving model ...
Updating learning rate to 3.125e-06
100%|█████████████████████████████████████████| 569/569 [12:45<00:00,  1.35s/it]
Epoch 7: Train Loss 0.313484 | Val Loss 4.040599 | Test Loss 4.064041
Validation loss decreased (4.128956 --> 4.040599). Saving model ...
Updating learning rate to 1.5625e-06
100%|█████████████████████████████████████████| 569/569 [12:45<00:00,  1.35s/it]
Epoch 8: Train Loss 0.313508 | Val Loss 4.078683 | Test Loss 4.000610
EarlyStopping counter: 1 out of 3
Updating learning rate to 7.8125e-07
100%|█████████████████████████████████████████| 569/569 [12:54<00:00,  1.36s/it]
Epoch 9: Train Loss 0.312852 | Val Loss 4.136350 | Test Loss 4.133328
EarlyStopping counter: 2 out of 3
Updating learning rate to 3.90625e-07
100%|█████████████████████████████████████████| 569/569 [12:48<00:00,  1.35s/it]
Epoch 10: Train Loss 0.312460 | Val Loss 4.153483 | Test Loss 4.148024
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>> testing : ECL_Autoformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 4.064704418182373, MAE: 1.6983731985092163, RMSE: 2.016111135482788, MAPE: 13.925434112548828, MSPE: 4799858.5
Args in experiment:
Namespace(is_training=1, task_id='traffic', model='Autoformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/traffic/', data_path='traffic.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=862, dec_in=862, c_out=862, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=3, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
AutoCorrelation used!
AutoCorrelation used!
AutoCorrelation used!
AutoCorrelation used!
>>>>>>> start training : traffic_Autoformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 377/377 [10:11<00:00,  1.62s/it]
Epoch 1: Train Loss 0.661575 | Val Loss nan | Test Loss nan
Validation loss decreased (inf --> nan). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 377/377 [10:11<00:00,  1.62s/it]
Epoch 2: Train Loss 0.565770 | Val Loss 5.268661 | Test Loss 5.347694
Validation loss decreased (nan --> 5.268661). Saving model ...
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 377/377 [10:13<00:00,  1.63s/it]
Epoch 3: Train Loss 0.536242 | Val Loss 4.973202 | Test Loss 5.152693
Validation loss decreased (5.268661 --> 4.973202). Saving model ...
Updating learning rate to 2.5e-05
>>>>>>> testing : traffic_Autoformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 5.130242824554443, MAE: 1.940636157989502, RMSE: 2.2650039196014404, MAPE: 25.637948989868164, MSPE: 16973440.0
```

#### Log for FEDformer

```bash
Args in experiment:
Namespace(is_training=1, task_id='Exchange', model='FEDformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/exchange_rate/', data_path='exchange_rate.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=8, dec_in=8, c_out=8, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
FourierBlock used!
FourierBlock used!
FourierCrossAttention used!
>>>>>>> start training : Exchange_FEDformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 160/160 [20:43<00:00,  7.77s/it]
Epoch 1: Train Loss 0.231613 | Val Loss 0.186537 | Test Loss 0.153534
Validation loss decreased (inf --> 0.186537). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 160/160 [20:27<00:00,  7.67s/it]
Epoch 2: Train Loss 0.214900 | Val Loss 0.183794 | Test Loss 0.148823
Validation loss decreased (0.186537 --> 0.183794). Saving model ...
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 160/160 [20:22<00:00,  7.64s/it]
Epoch 3: Train Loss 0.211968 | Val Loss 0.184629 | Test Loss 0.147067
EarlyStopping counter: 1 out of 3
Updating learning rate to 2.5e-05
100%|█████████████████████████████████████████| 160/160 [20:22<00:00,  7.64s/it]
Epoch 4: Train Loss 0.210454 | Val Loss 0.184617 | Test Loss 0.146944
EarlyStopping counter: 2 out of 3
Updating learning rate to 1.25e-05
100%|█████████████████████████████████████████| 160/160 [19:56<00:00,  7.48s/it]
Epoch 5: Train Loss 0.209890 | Val Loss 0.184621 | Test Loss 0.146946
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>> testing : Exchange_FEDformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 0.14878332614898682, MAE: 0.278480589389801, RMSE: 0.38572442531585693, MAPE: 1.6986275911331177, MSPE: 1674.3245849609375
Args in experiment:
Namespace(is_training=1, task_id='weather', model='FEDformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/weather/', data_path='weather.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=21, dec_in=21, c_out=21, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
FourierBlock used!
FourierBlock used!
FourierCrossAttention used!
>>>>>>> start training : weather_FEDformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████| 1146/1146 [2:25:59<00:00,  7.64s/it]
Epoch 1: Train Loss 0.582818 | Val Loss 0.529124 | Test Loss 0.266590
Validation loss decreased (inf --> 0.529124). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████| 1146/1146 [2:23:05<00:00,  7.49s/it]
Epoch 2: Train Loss 0.542343 | Val Loss 0.527041 | Test Loss 0.266302
Validation loss decreased (0.529124 --> 0.527041). Saving model ...
Updating learning rate to 5e-05
100%|█████████████████████████████████████| 1146/1146 [2:21:47<00:00,  7.42s/it]
Epoch 3: Train Loss 0.537212 | Val Loss 0.522657 | Test Loss 0.254703
Validation loss decreased (0.527041 --> 0.522657). Saving model ...
Updating learning rate to 2.5e-05
100%|█████████████████████████████████████| 1146/1146 [2:21:37<00:00,  7.41s/it]
Epoch 4: Train Loss 0.536247 | Val Loss 0.521688 | Test Loss 0.250680
Validation loss decreased (0.522657 --> 0.521688). Saving model ...
Updating learning rate to 1.25e-05
100%|█████████████████████████████████████| 1146/1146 [2:21:41<00:00,  7.42s/it]
Epoch 5: Train Loss 0.537666 | Val Loss 0.522536 | Test Loss 0.251573
EarlyStopping counter: 1 out of 3
Updating learning rate to 6.25e-06
100%|█████████████████████████████████████| 1146/1146 [2:21:30<00:00,  7.41s/it]
Epoch 6: Train Loss 0.539238 | Val Loss 0.524007 | Test Loss 0.252426
EarlyStopping counter: 2 out of 3
Updating learning rate to 3.125e-06
100%|█████████████████████████████████████| 1146/1146 [2:21:32<00:00,  7.41s/it]
Epoch 7: Train Loss 0.540547 | Val Loss 0.523277 | Test Loss 0.248727
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>> testing : weather_FEDformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 0.2510143518447876, MAE: 0.3188607096672058, RMSE: 0.5010133385658264, MAPE: 18.864458084106445, MSPE: 40260700.0
Args in experiment:
Namespace(is_training=1, task_id='ECL', model='FEDformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/electricity/', data_path='electricity.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=321, dec_in=321, c_out=321, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
FourierBlock used!
FourierBlock used!
FourierCrossAttention used!
>>>>>>> start training : ECL_FEDformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|███████████████████████████████████████| 569/569 [1:13:03<00:00,  7.70s/it]
Epoch 1: Train Loss 0.361673 | Val Loss 3.072628 | Test Loss 3.096645
Validation loss decreased (inf --> 3.072628). Saving model ...
Updating learning rate to 0.0001
100%|███████████████████████████████████████| 569/569 [1:12:07<00:00,  7.61s/it]
Epoch 2: Train Loss 0.281545 | Val Loss 3.060035 | Test Loss 3.092298
Validation loss decreased (3.072628 --> 3.060035). Saving model ...
Updating learning rate to 5e-05
100%|███████████████████████████████████████| 569/569 [1:11:11<00:00,  7.51s/it]
Epoch 3: Train Loss 0.278860 | Val Loss 2.824183 | Test Loss 2.843272
Validation loss decreased (3.060035 --> 2.824183). Saving model ...
Updating learning rate to 2.5e-05
100%|███████████████████████████████████████| 569/569 [1:11:15<00:00,  7.51s/it]
Epoch 4: Train Loss 0.282020 | Val Loss 2.645290 | Test Loss 2.646105
Validation loss decreased (2.824183 --> 2.645290). Saving model ...
Updating learning rate to 1.25e-05
100%|███████████████████████████████████████| 569/569 [1:11:18<00:00,  7.52s/it]
Epoch 5: Train Loss 0.285485 | Val Loss 2.560224 | Test Loss 2.543272
Validation loss decreased (2.645290 --> 2.560224). Saving model ...
Updating learning rate to 6.25e-06
100%|███████████████████████████████████████| 569/569 [1:11:26<00:00,  7.53s/it]
Epoch 6: Train Loss 0.287430 | Val Loss 2.462325 | Test Loss 2.440337
Validation loss decreased (2.560224 --> 2.462325). Saving model ...
Updating learning rate to 3.125e-06
100%|███████████████████████████████████████| 569/569 [1:11:33<00:00,  7.55s/it]
Epoch 7: Train Loss 0.287324 | Val Loss 2.462385 | Test Loss 2.438992
EarlyStopping counter: 1 out of 3
Updating learning rate to 1.5625e-06
100%|███████████████████████████████████████| 569/569 [1:11:25<00:00,  7.53s/it]
Epoch 8: Train Loss 0.286453 | Val Loss 2.485177 | Test Loss 2.461610
EarlyStopping counter: 2 out of 3
Updating learning rate to 7.8125e-07
100%|███████████████████████████████████████| 569/569 [1:11:36<00:00,  7.55s/it]
Epoch 9: Train Loss 0.285710 | Val Loss 2.483797 | Test Loss 2.460788
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>> testing : ECL_FEDformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 2.440469264984131, MAE: 1.3262627124786377, RMSE: 1.5622001886367798, MAPE: 10.921345710754395, MSPE: 2729983.5
Args in experiment:
Namespace(is_training=1, task_id='traffic', model='FEDformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh', data='custom', root_path='/home/ma-user/work/fedformer/dataset/traffic/', data_path='traffic.csv', features='M', target='OT', freq='h', detail_freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, enc_in=862, dec_in=862, c_out=862, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], factor=3, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, distil=True, num_workers=10, itr=1, train_epochs=3, batch_size=32, patience=3, learning_rate=0.0001, des='Exp', loss='mse', lradj='type1', use_gpu=True, gpu=0, use_multi_gpu=False, devices='0')
Using GPU: 0
FourierBlock used!
FourierBlock used!
FourierCrossAttention used!
>>>>>>> start training : traffic_FEDformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
100%|█████████████████████████████████████████| 377/377 [49:45<00:00,  7.92s/it]
Epoch 1: Train Loss 0.585562 | Val Loss 4.502412 | Test Loss 4.599408
Validation loss decreased (inf --> 4.502412). Saving model ...
Updating learning rate to 0.0001
100%|█████████████████████████████████████████| 377/377 [48:27<00:00,  7.71s/it]
Epoch 2: Train Loss 0.469448 | Val Loss 4.610301 | Test Loss 4.691549
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
100%|█████████████████████████████████████████| 377/377 [48:17<00:00,  7.68s/it]
Epoch 3: Train Loss 0.437180 | Val Loss 4.345877 | Test Loss 4.416132
Validation loss decreased (4.502412 --> 4.345877). Saving model ...
Updating learning rate to 2.5e-05
>>>>>>> testing : traffic_FEDformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp
Test result - MSE: 4.412840366363525, MAE: 1.7953130006790161, RMSE: 2.1006760597229004, MAPE: 24.045167922973633, MSPE: 21861018.0
```

