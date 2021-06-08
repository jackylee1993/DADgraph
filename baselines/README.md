# Baselines

### Classical MRC models for document MRC. 
We adopt open-source code of the following two models from the SQuAD 2.0 paper: https://bit.ly/2rDHBgY
* BiDAF. The BiDAF model presents the context passage at different levels of granularity and learns the query-aware context representation using a bi-directional attention flow mechanism.
* DocQA. This model is a neural paragraph-level QA method, which can scale to document and multidocument inputs. DocQA can ignore no-answer containing paragraphs in documents. The model contains paragraph sampling and attempts to produce a globally correct answer.

### Popular pre-trained model
We adopt the open-source code of BERT: https://github.com/google-research/bert
* BERT. To adapt BERT for our task, we concatenate all utterances from the input dialogue as a passage, where each utterance ui encodes both the speaker identity and their uttered
text as {speaker_ui : content_ui}.

### Two dialogue models: [DialogueRNN](https://ojs.aaai.org/index.php/AAAI/article/view/4657) and [DialogueGCN](https://www.aclweb.org/anthology/D19-1015/) with MRC module.

We adopt DialogueRNN and DialogueGCN as our baselines. These two models are originally designed for sentiment classification. To adapt them to our task, we replace DADgraph’s internal models with these models but hold fixed the same final MRC module and BERT-based utterance representations.

**DialogueRNN**. DialogueRNN is a sequential neural network model for representing multiparty dialogues on emotion recognition for conversations task with two bidirectional
GRUs: a global GRU and a party GRU.

**DialogueGCN**. Compared to DialogueRNN, DialogueGCN model the context windows of an utterance in the dialogue as a graph and represent the graph using the GCN model.
