import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np, itertools, random, copy, math
from transformers import *

class UnstructuredEncoder(nn.Module):
#output : input_dim,config,output_dim
    def __init__(self,output_dim):
        super(UnstructuredEncoder, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (BertModel,       BertTokenizer,       'bert-base-uncased')
        self.model = model_class.from_pretrained(pretrained_weights)
        self.qa_outputs = nn.Linear(self.model.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.model(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        return self.qa_outputs(outputs[0])  # start_logits, end_logits, (hidden_states), (attentions)
