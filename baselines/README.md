## Two baselines: [DialogueRNN](https://ojs.aaai.org/index.php/AAAI/article/view/4657) and [DialogueGCN](https://www.aclweb.org/anthology/D19-1015/) for MRC module.

We adopt DialogueRNN and DialogueGCN as our baselines. These two models are originally designed for sentiment classification. To adapt them to our task, we replace DADgraph’s internal models with these models, but hold fixed the same final MRC module and BERT-based utterance representations.

**DialogueRNN**. DialogueRNN is a sequential neural network model for representing multiparty dialogues on emotion recognition for conversations task with two bidirectional
GRUs: a global GRU and a party GRU.

**DialogueGCN**. Compared to DialogueRNN, DialogueGCN model the context windows of an utterance in the dialogue as a graph and represent the graph using the GCN model.
