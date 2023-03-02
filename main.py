"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

import torch_npu.npu
from munch import Munch
from torch.backends import cudnn
import torch
import torch.multiprocessing as mp

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def distribute_run(npu, in_args):
    print(f'---- distribute npu {npu} ----')
    args = in_args
    solver = Solver(args, npu)
    loaders = Munch(src=get_train_loader(distribute=args.distribute,
                                         root=args.train_img_dir,
                                         which='source',
                                         img_size=args.img_size,
                                         batch_size=args.batch_size,
                                         prob=args.randcrop_prob,
                                         num_workers=args.num_workers),
                    ref=get_train_loader(distribute=args.distribute,
                                         root=args.train_img_dir,
                                         which='reference',
                                         img_size=args.img_size,
                                         batch_size=args.batch_size,
                                         prob=args.randcrop_prob,
                                         num_workers=args.num_workers),
                    val=get_test_loader(distribute=args.distribute,
                                        root=args.val_img_dir,
                                        img_size=args.img_size,
                                        batch_size=args.val_batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers))
    solver.train(loaders)


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains

        if args.distribute:
            # 单机多卡
            mp.spawn(distribute_run, nprocs=args.world_size, args=(args,))
        else:
            # 单机单卡
            solver = Solver(args, 0)
            loaders = Munch(src=get_train_loader(distribute=args.distribute,
                                                 root=args.train_img_dir,
                                                 which='source',
                                                 img_size=args.img_size,
                                                 batch_size=args.batch_size,
                                                 prob=args.randcrop_prob,
                                                 num_workers=args.num_workers),
                            ref=get_train_loader(distribute=args.distribute,
                                                 root=args.train_img_dir,
                                                 which='reference',
                                                 img_size=args.img_size,
                                                 batch_size=args.batch_size,
                                                 prob=args.randcrop_prob,
                                                 num_workers=args.num_workers),
                            val=get_test_loader(distribute=args.distribute,
                                                root=args.val_img_dir,
                                                img_size=args.img_size,
                                                batch_size=args.val_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers))
            solver.train(loaders)
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))

        solver = Solver(args, 0)
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver = Solver(args, 0)
        solver.evaluate()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=1)  # save every epoch 每隔多少轮次保存一次
    parser.add_argument('--eval_every', type=int, default=50000)


    # ascend dist argument
    # parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
    parser.add_argument('--npu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--nnode', type=int, default=1,
                        help='distributed training machine node nums. single is 1')  # 分布式训练节点机器数
    parser.add_argument('--nproc_per_node', type=int, default=1,
                        help='distributed training machine processes per node')  # 分布式训练单节点进程数（相当于npu数量）
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of processes for distributed training')  # 分布式训练全局进程数

    parser.add_argument('--amp', default=False, action='store_true', help='use amp to train the model')  # 开启混合精度
    parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')
    parser.add_argument('--distribute', default=False, action='store_true', help='distribute training')  # 开启分布式训练

    parser.add_argument('--dist_backend', default='hccl', type=str,
                        help='distributed backend')

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'
    if args.world_size == -1:
        args.world_size = args.nnode * torch_npu.npu.device_count()
    args.npu_ddp = args.world_size > 1

    main(args)
