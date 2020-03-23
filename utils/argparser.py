import argparse
import os
import torch

def argparser(batch_size=128, epochs=1000, seed=0, verbose=1, lr=1e-4,
              opt='adam', momentum=0.9, weight_decay=5e-4):

    parser = argparse.ArgumentParser()

    # optimizer settings, shared between DIM and classification
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument('--gpu', action='store_true')

    # DIM arguments, note that global_dim is also used to select the encoder used for classification
    parser.add_argument('--global_dim', action="store_true")
    parser.add_argument('--nce', action='store_true')

    # Classifier arguments
    parser.add_argument('--input_layer', default='fc')
    parser.add_argument('--hidden_units', type=int, default=1024)
    parser.add_argument('--encoder_ckpt', type=str, default=None)

    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None)

    args = parser.parse_args()
    if args.cuda_ids is not None:
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args
