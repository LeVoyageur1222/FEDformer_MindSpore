import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import argparse
import os
import random
import numpy as np
import mindspore as ms
from exp.exp_main_ms import Exp_Main

def main():
    # 设置随机种子
    ms.set_seed(2025)
    random.seed(2025)
    np.random.seed(2025)

    parser = argparse.ArgumentParser(description='MindSpore Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--task_id', type=str, default='test')
    parser.add_argument('--model', type=str, default='FEDformer',
                        help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # FEDformer config
    parser.add_argument('--version', type=str, default='Fourier')
    # parser.add_argument('--version', type=str, default='Wavelets')
    parser.add_argument('--mode_select', type=str, default='random')
    parser.add_argument('--modes', type=int, default=64)
    parser.add_argument('--L', type=int, default=3)
    parser.add_argument('--base', type=str, default='legendre')
    parser.add_argument('--cross_activation', type=str, default='tanh')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--detail_freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # model define
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', default=[24], type=int)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type1')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--devices', type=str, default='0')

    args = parser.parse_args()

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.task_id, args.model, args.mode_select, args.modes, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
                args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.distil, args.des, ii
            )

            exp = Exp(args)
            print(f'>>>>>>> start training : {setting}')
            exp.train(setting)

            print(f'>>>>>>> testing : {setting}')
            exp.test(setting)

            if args.do_predict:
                print(f'>>>>>>> predicting : {setting}')
                exp.predict(setting, load=True)

    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.task_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.distil, args.des, ii
        )
        exp = Exp(args)
        print(f'>>>>>>> testing : {setting}')
        exp.test(setting, test=1)

if __name__ == '__main__':
    main()
