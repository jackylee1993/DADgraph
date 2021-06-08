# [DADgraph: A Discourse-aware Dialogue Graph Neural Network for Multiparty Dialogue Machine Reading Comprehension](https://arxiv.org/abs/2104.12377)

This is the code of DADgraph paper, accepted by the International Joint Conference on Neural Networks(IJCNN), 2021. 

* Authors: Jiaqi Li, Ming Liu, Zihao Zheng, Heng Zhang, Bing Qin, Min-Yen Kan, Ting Liu. 

* Abstract: Multiparty Dialogue Machine Reading Comprehension (MRC) differs from traditional MRC as models must handle the complex dialogue discourse structure, previously unconsidered in traditional MRC. To fully exploit such discourse structure in multiparty dialogue, we present a discourse-aware dialogue graph neural network, DADgraph, which explicitly constructs the dialogue graph using discourse dependency links and discourse relations. To validate our model, we perform experiments on the Molweni corpus, a large-scale MRC dataset built over multiparty dialogue annotated with discourse structure. Experiments on Molweni show that our discourse-aware model achieves statistically significant improvements compared against strong neural network MRC baselines.

* Dataset: [Molweni corpus](https://github.com/hit-scir/molweni).


## Dependencies

* Python 3
* PyTorch 

## Usage

python train.py
