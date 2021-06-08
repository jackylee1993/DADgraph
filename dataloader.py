import torch
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import os
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)
import torch
from transformers import *
model_class, tokenizer_class, pretrained_weights = (BertModel,       BertTokenizer,       'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights).cuda()
d_h = 100 #输入向量的维度
linear = torch.nn.Linear(model.config.hidden_size,d_h)

def get_utterance_reps(text):#获取每个句子单独的表示
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).cuda()
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

    return linear(last_hidden_states[0][0])


def loadMolweni(mode,args):
    dpmode =open("./Molweni/DP/"+mode+".json")
    dpdata =json.loads(dpmode.read())
    dataset_raw, example_num = load_and_cache_examples(tokenizer, mode=mode,args = args)
    mrcdataset = []
    qas_start = 0
    for i in example_num:
        qas_start += i
        mrcdataset.append(dataset_raw[qas_start-i:qas_start])
    return dpdata,mrcdataset



def load_and_cache_examples(tokenizer,mode = None,args=None):
    # Load data features from cache or dataset file
    input_file = "./Molweni/MRC/" + mode + ".json"
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    cached_features_file = os.path.join(os.path.dirname(input_file), mode+'cached')


    if os.path.exists(cached_features_file):
        cached_numbers_file = open(os.path.join(os.path.dirname(input_file), mode + 'cached_num'), 'rb')
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        example_num = pickle.load(cached_numbers_file)
    else:
        examples,example_num = read_squad_examples(input_file=input_file,
                                                is_training=True,
                                                version_2_with_negative=True)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=384,
                                                doc_stride=128,
                                                max_query_length=64,
                                                is_training=True,
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=False,   # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            cached_numbers_file = open(os.path.join(os.path.dirname(input_file), mode + 'cached_num'), 'wb')
            pickle.dump(example_num,cached_numbers_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_start_positions, all_end_positions,
                            all_cls_index, all_p_mask)
    cached_numbers_file.close()

    return dataset,example_num

class MolweniDataset(Dataset):
    def __init__(self,mode=None,args=None):
        #此处修改一下 需要将数据集格式处理成IEMO格式
        self.dP,self.mrcdataset= loadMolweni(mode,args = args)


    def __getitem__(self, index):
        #textf dialoglen,dim
        edus = []
        speakers = set()
        for edu in self.dP[index]['edus']:
            edus.append(get_utterance_reps(edu['text']))
            speakers.add(edu['speaker'])
        textf = torch.stack(edus)
        speakers = list(speakers)
        speakers_num = len(speakers)
        qmask = []
        for edu in self.dP[index]['edus']:
            tmp = torch.zeros(speakers_num)
            tmp[speakers.index(edu['speaker'])] = 1
            qmask.append(tmp)
        qmask = torch.stack(qmask)
        umask = torch.FloatTensor([1]*len(self.dP[index]['edus']))
        return textf,qmask,umask,self.mrcdataset[index],self.dP[index]['relations']

    def __len__(self):
        return len(self.dP)

def get_Molweni_loaders(batch_size = 1,num_workers = 0,pin_memory = False,args=None):
    trainset =MolweniDataset('train',args = args)
    valset =MolweniDataset('dev',args = args)
    testset = MolweniDataset('test',args = args)
    train_loader =DataLoader(
        trainset,
        batch_size = batch_size,
        sampler = get_sampler(trainset),
        collate_fn= lambda x:x,
        num_workers= num_workers,
        pin_memory= pin_memory
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        sampler=get_sampler(valset),
        collate_fn=lambda x: x,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        sampler=get_sampler(testset),
        collate_fn=lambda x: x,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader,val_loader,test_loader

def get_sampler(set):
    size = len(set)
    idx =list(range(size))
    return SubsetRandomSampler(idx)



if __name__ == "__main__":
    testset = MolweniDataset('test')
    test_loader = DataLoader(
        testset,
        batch_size=32,
        sampler=get_sampler(testset),
        collate_fn=lambda x:x,
        num_workers=0,
        pin_memory=False
    )
    for data in test_loader:
        print(data)

