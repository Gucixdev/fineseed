# sparsellm (verified: arXiv 2402.17946, NeurIPS 2024, github.com/BaiTheBest/SparseLLM Apache-2.0)

What it is: a global PRUNING method for language models. Lossy, changes the model; it is not a storage format.
Its only link to fineseed: a pruned model has many zeros, which a coder must store cheaply. The sparse-storage side is
another paper (SpInfer: a bitmap instead of indices, since indices can cancel the savings near 50% sparsity).

Take: nothing now. Later, if pruned models are packed: test a bitmap + non-zero values plane against zstd on zeros.
