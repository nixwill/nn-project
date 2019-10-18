# Project proposal

## Motivation

We will be solving the task of machine translation between natural languages.
It is arguably the most widely applied and researched natural language processing task in practice today.

Translating long sentences is one of the common problems that neural machine translation models can be facing.
When using a recurrent architecture, the model often tends to forget the beginning of the sentence before it finishes processing it to the end.
This has been first partly solved by the invention of the LSTM and the GRU, both of which implements a forget gate – a part of the cell which selectively drops remembered information that it deems not useful, so it can remember more of the necessary information.

A popular method for translating sentences today is transforming the input sequence into a feature vector using an encoder network and then decoding it into a sentence in another language using another network.
However this solution is far from perfect for translating very long sentences, as the entire sentence has to be embedded into a single fixed-length vector.
Currently, many of the state-of-the-art methods implement an attention mechanism which trains the model to pay selective attention to each step of the input sequence [[source]](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/).


## Related work

Proposal of the encoder–decoder model:

* [Sutskever, I. et al.: Sequence to Sequence Learning with Neural Networks. *arXiv*. 2014.](https://arxiv.org/pdf/1706.03762v5.pdf)

Proposal of the attention mechanism:

* [Bahdanau, D. et al.: Neural Machine Translation by Jointly Learning to Align and Translate. *arXiv*. 2017.](https://arxiv.org/pdf/1706.03762v5.pdf)

Proposal of the Transformer – a model that completely abandons recurrence and convolution in favor of attention:

* [VASWANI, A. et al.: Attention Is All You Need. *arXiv*. 2017.](https://arxiv.org/pdf/1706.03762v5.pdf)


## Dataset

We will be using the [_WMT'15_ English–Czech dataset](https://nlp.stanford.edu/projects/nmt/) provided by the _The Stanford Natural Language Processing Group_.

It contains 15,794,564 pairs of tokenized sentences in English and Czech.


## Proposed solution

We will attempt to create a solution using a recurrent encoder–decoder model.

As an experiment, we will compare different recurrent architectures, such as a standard RNN and an LSTM or a GRU, against a baseline solution.
