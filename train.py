import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import get_Molweni_loaders
from dataloader import linear
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from StructuredEncoder import OurDialogueGCNModel
from UnstructuredEncoder import UnstructuredEncoder
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
from StructuredEncoder import Attention
from tqdm import tqdm, trange
from transformers import AdamW
# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_resemble_embedding(structuredEmbed,unstructuredEmbed,attn):
    structuredEmbed = structuredEmbed[0].permute(1,0,2).repeat(unstructuredEmbed.shape[0],1,1)
    output,_ = attn(structuredEmbed,unstructuredEmbed)
    print(output.shape)
    output = torch.cat((output,unstructuredEmbed),dim = -1)
    return output

def mrclossfunc(start_positions,end_positions,logits):
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
    return total_loss


if __name__ == '__main__':

    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', type = int, default=3,
                        help='DialogueRNN 1 DialogueGCN 2 DADgraph 3')

    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")


    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    D_m = 100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100
    QA_output = nn.Linear(2 * graph_h, 2)
    structuredModel = None
    print(args.graph_model)
    if args.graph_model is 1:
        seed_everything()  # 初始化种子，目的是获得更好的随机数
        structuredModel = DialogueGCNModel(args.base_model,  # 定义模型
                                 D_m, D_g, D_p, D_e, D_h, D_a, D_h,
                                 n_speakers=10,
                                 max_seq_len=110,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=10,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda)

        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    elif args.graph_model is 2:
        structuredModel = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a,
                               n_classes=7,
                               listener_state=args.active_listener,
                               context_attention=args.attention,
                               dropout_rec=args.rec_dropout,
                               dropout=args.dropout)

        print('Basic Dialog RNN Model.')
        name = 'Base'
    else:
        structuredModel = OurDialogueGCNModel(args.base_model, D_m,  D_e, graph_h,  110,16,
                  dropout=args.dropout, nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda)

    resemble_attn = Attention(embed_dim = graph_h,hidden_dim = graph_h,out_dim = graph_h, score_function='mlp')
    if cuda:
        structuredModel.cuda()
        QA_output.cuda()
        resemble_attn.cuda()
        linear.cuda()

    unstructuredModel = UnstructuredEncoder(graph_h)
    if cuda:
        unstructuredModel.cuda()



    train_loader, valid_loader, test_loader = get_Molweni_loaders(args = args)
    model_parameters = []
    model_parameters += unstructuredModel.parameters()
    model_parameters += structuredModel.parameters()
    model_parameters += QA_output.parameters()
    model_parameters +=resemble_attn.parameters()
    model_parameters += linear.parameters()
    optimizer = Adam(model_parameters,lr = 0.0001)

    for e in range(n_epochs):
        seed_everything()
        for datas in train_loader:

            loss = torch.tensor([0], dtype=torch.float, requires_grad=True)
            if cuda:
                loss = loss.cuda()
            for data in datas:
                #todo:此处需要一个optimizer
                # umask是对话的数目
                textf, qmask, umask, mrc,edges = data
                textf = textf.unsqueeze(1)
                qmask = qmask.unsqueeze(1)
                umask = umask.unsqueeze(0)
                if cuda:
                    textf = textf.cuda()
                    qmask = qmask.cuda()
                    umask = umask.cuda()
                lengths = [len(umask.reshape(-1))]
                if args.graph_model is 1:
                    structuredEmbed = structuredModel(textf, qmask, umask, lengths)
                elif args.graph_model is 2:
                    structuredEmbed =structuredModel(textf, qmask, umask)
                else:
                    structuredEmbed = structuredModel(textf, umask, lengths,edges)
                batch = tuple(t.to(torch.device('cuda')) if args.cuda else t.to(torch.device('cpu')) for t in mrc)
                # print(type(batch))
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] ,
                          'start_positions': batch[3],
                          'end_positions': batch[4]}
                unstructuredEmbed = unstructuredModel(**inputs)
                # todo:将有结构和无结构向量拼接为sequence_output
                sequence_output = get_resemble_embedding(structuredEmbed,unstructuredEmbed,resemble_attn)
                logits = QA_output(sequence_output)
                if logits.shape[0] is 0:
                    continue
                oneloss = mrclossfunc(inputs['start_positions'],inputs['end_positions'],logits)
                loss += oneloss/batch[3].size(0)
            optimizer.zero_grad()
            loss = loss/batch_size
            print("loss:",loss.detach())
            loss.backward()
            optimizer.step()


