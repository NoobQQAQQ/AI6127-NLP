export DATA_DIR=$1
export TASK_NAME=$2
export MODEL_NAME=albert-xxlarge-v2

CUDA_VISIBLE_DEVICES=0 python run_multiple_choice.py \
    --model_type albert \
    --model_name_or_path $MODEL_NAME \
    --cache_dir ../cache \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 2  \
    --per_gpu_train_batch_size 2  \
    --gradient_accumulation_steps 12 \
    --learning_rate 1e-5 \
    --num_train_epochs 10.0 \
    --output_dir ../Checkpoints/$TASK_NAME/${MODEL_NAME} \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
