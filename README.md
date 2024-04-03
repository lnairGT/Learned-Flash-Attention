## Can FlashAttention Statistics be Learned?
This is a small project that I worked on that tries to learn the statistics (`m(x)` and `l(x)`) computed in [FlashAttention](https://github.com/Dao-AILab/flash-attention). The basic idea is as follows:
$softmax([A_1, A_2, ... A_N]) = [\alpha_1 * softmax(A_1), \alpha_2 * softmax(A_2), ... \alpha_N * softmax(A_N)]$
Here, $\alpha_i \in \mathcal{R}^{S}$, where $S$ is the sequence length. This allows the softmax operation to be performed in blocks, enabling sequence parallelism -- i.e., the operation $softmax(QK^T)$ can be performed in independent blocks (hence processed in parallel). The original FlashAttention paper computes the statistics over each iteration of a for loop, as each block of Q and K is loaded. In this project, I try to learn these statistics (or scales) $\alpha_1 ... \alpha_N$.

**NOTE:** During training, _all_ layers of the model are _frozen_ and _only_ the scales are learned.

### Running the code

#### Requirements
This repo requires [Pytorch](https://github.com/pytorch/pytorch) and HuggingFace [Transformers](https://github.com/huggingface/transformers) version **4.27.4**. This code has not been tested on the more recent versions of HF Transformers.

The scripts are adapted from HuggingFace. A sample script for running OPT-350M, dividing softmax into two blocks is shown in `run_opt_train.sh`.
```
python run_clm_no_trainer.py \
    --model_name_or_path lnair/opt-350m-wikitext2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --output_dir opt-350m-softmax-scales \
    --checkpointing_steps epoch \
    --num_train_epochs 30 \
    --scales_lr 1e-2 \
    --num_softmax_blocks 2
```

### Results on OPT-Models
The following PPL values are obtained on Wikitext-2, with the learned scales. The following arguments are used:
```
--num_softmax_blocks 2
--scales_lr 1e-2
--num_train_epochs 50
```

| Model    | PPL (FP32) | PPL (Ours) |
| -------- | ---------- | ---------- |
| OPT-125M | 26         | 22.36      |
| OPT-350M | 16.84      | 17.81      |
| OPT-1.3B | 11.78      | 12.63      |
| OPT-2.7B | 11         | 12.56      |

#### Visualizations
Comparisons of generated attention maps and spikiness (For reference on what "spikiness" means, please see [this](https://arxiv.org/abs/2402.04347) paper on Softmax mimicry). The plots titled "no trained statistics" refers to cases where the softmax is broken into blocks without applying any correction with the scales. When the learned scales are applied, the softmax outputs match the ground truth baseline.

![alt text](assets/attention_weights.png)
![alt text](assets/Spikiness.png)
