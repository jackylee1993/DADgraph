# [DADgraph: A Discourse-aware Dialogue Graph Neural Network for Multiparty Dialogue Machine Reading Comprehension](https://arxiv.org/abs/2104.12377)

This is the code of DADgraph paper, accepted by the International Joint Conference on Neural Networks(IJCNN), 2021. 

* Authors: Jiaqi Li, Ming Liu, Zihao Zheng, Heng Zhang, Bing Qin, Min-Yen Kan, Ting Liu. 

* Abstract: Multiparty Dialogue Machine Reading Comprehension (MRC) differs from traditional MRC as models must handle the complex dialogue discourse structure, previously unconsidered in traditional MRC. To fully exploit such discourse structure in multiparty dialogue, we present a discourse-aware dialogue graph neural network, DADgraph, which explicitly constructs the dialogue graph using discourse dependency links and discourse relations. To validate our model, we perform experiments on the Molweni corpus, a large-scale MRC dataset built over multiparty dialogue annotated with discourse structure. Experiments on Molweni show that our discourse-aware model achieves statistically significant improvements compared against strong neural network MRC baselines.

* Dataset: [Molweni corpus](https://github.com/hit-scir/molweni).

* Cite
@INPROCEEDINGS{9533364,
  author={Li, Jiaqi and Liu, Ming and Zheng, Zihao and Zhang, Heng and Qin, Bing and Kan, Min-Yen and Liu, Ting},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)}, 
  title={DADgraph: A Discourse-aware Dialogue Graph Neural Network for Multiparty Dialogue Machine Reading Comprehension}, 
  year={2021},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN52387.2021.9533364}}


## Dependencies

* Python 3
* PyTorch 
* [Transformers 3.4](https://github.com/huggingface/transformers)
* [BERT-base, uncased model](https://huggingface.co/bert-base-uncased/tree/main)

## Descriptions

`train.py`: main file, including three models:
1. DialogueRNN
2. DialogueGCN
3. DADgraph

Models can be choosed by `graph-model` parameter.

## Acknowledgements

We referenced codes of [DialogueGCN (EMNLP'19)](https://github.com/declare-lab/conv-emotion#dialoguegcn-a-graph-convolutional-neural-network-for-emotion-recognition-in-conversation) and [DeepSequential (AAAI'19)](https://github.com/shizhouxing/DialogueDiscourseParsing).
