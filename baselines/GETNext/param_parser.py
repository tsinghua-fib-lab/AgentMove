"""Parsing the parameters."""
import argparse

import torch

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GETNext.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')
    # parser.add_argument('--device',
    #                     type=str,
    #                     default=device,
    #                     help='')
    # Data
    parser.add_argument('--dataset_name',
                    type=str,
                    default='NYC',
                    help='Name of dataset')
    parser.add_argument('--train_sample',
                type=float,
                choices=[0.3,0.1,0.5, 0.7])
    parser.add_argument('--data_adj_mtx',
                        type=str,
                        default='graph_A.npy',
                        help='Graph adjacent path')
    parser.add_argument('--data_node_feats',
                        type=str,
                        default='graph_X.npy',
                        help='Graph node features path')
    parser.add_argument('--data_train',
                        type=str,
                        default='train.csv',
                        help='Training data path')
    parser.add_argument('--data_test',
                        type=str,
                        default='test.csv',
                        help='Test data path')
    parser.add_argument('--short_traj_thres',
                        type=int,
                        default=2,
                        help='Remove over-short trajectory')
    parser.add_argument('--time_units',
                        type=int,
                        default=48,
                        help='Time unit is 0.5 hour, 24/0.5=48')
    parser.add_argument('--time_feature',
                        type=str,
                        default='norm_in_day_time',
                        help='The name of time feature in the data')

    # Model hyper-parameters
    parser.add_argument('--poi-embed-dim',
                        type=int,
                        default=128,
                        help='POI embedding dimensions')
    parser.add_argument('--user-embed-dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid',
                        type=list,
                        default=[32, 64],
                        help='List of hidden dims for gcn layers')
    parser.add_argument('--transformer-nhid',
                        type=int,
                        default=1024,
                        help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers',
                        type=int,
                        default=2,
                        help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead',
                        type=int,
                        default=2,
                        help='Num of heads in multiheadattention')
    parser.add_argument('--transformer-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for transformer')
    parser.add_argument('--time-embed-dim',
                        type=int,
                        default=32,
                        help='Time embedding dimensions')
    parser.add_argument('--cat-embed-dim',
                        type=int,
                        default=32,
                        help='Category embedding dimensions')
    parser.add_argument('--time-loss-weight',
                        type=int,
                        default=10,
                        help='Scale factor for the time loss term')
    parser.add_argument('--node-attn-nhid',
                        type=int,
                        default=128,
                        help='Node attn map hidden dimensions')

    # Training hyper-parameters
    parser.add_argument('--batch',
                        type=int,
                        default=20,
                        help='Batch size.')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=False,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--project',
                        default='runs/train',
                        help='save to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--mode',
                        type=str,
                        default='client',
                        help='python console use only')
    parser.add_argument('--port',
                        type=int,
                        default=64973,
                        help='python console use only')
    parser.add_argument('--gpu_id',
                    type=int,
                    default=-1,
                    help='GPU ID to use, -1 for CPU only')
    
    args = parser.parse_args()

    # GPU 设置
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using default GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available, using CPU instead.")

    args.device = device
    return args


