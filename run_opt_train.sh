#MODEL_NAME="Aalaa/opt-125m-wikitext2"
MODEL_NAME="lnair/opt-350m-wikitext2"
#MODEL_NAME="lnair/opt-1.3b-wikitext2"
#MODEL_NAME="lnair/opt-2.7b-wikitext2"

python run_clm_no_trainer.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --output_dir opt-350m-softmax-scales \
    --checkpointing_steps epoch \
    --num_train_epochs 30 \
    --scales_lr 1e-2 \
    --num_softmax_blocks 2
    #--block_size 512
