To run the code, you can simply use the following command:

```bash
bash BASH_FILE DATA_DIR TASK_NAME
```

There are 4 bash files, corresponding to BERT, ALBERT, RoBERTa and XLNet respectively.
The `DATA_DIR` refers to the directory of the dataset.
The `TASK_NAME` refers to the name of the task to run.

For example, you can use the the following command to run BERT on ReClor:

```bash
bash run_bert.sh ../data/ReClor reclor
```


Besides, you can assign different values to arguments in the bash file to customize your running.
The script accepts the following arguments:

```bash
optional arguments:
    --data_dir                      The input data dir contains the data files for the task
    --model_type                    Model type selected in the list: bert, xlnet, roberta, albert
    --model_name_or_path            Path to pre-trained model or shortcut name
    --task_name                     The name of the task to train
    --cache_dir                     Where do you want to store the pre-trained models downloaded from Internet
    --output_dir                    The output directory where the model predictions and checkpoints will be written
    --do_train                      Whether to run training
    --do_eval                       Whether to run eval on the dev set
    --do_test                       Whether to run test on the test set
    --evaluate_during_training      Run evaluation during training at each logging step
    --do_lower_case                 Set this flag if you are using an uncased model
    --max_seq_length                The maximum total input sequence length after tokenization
    --per_gpu_eval_batch_size       Batch size per GPU for evaluation
    --per_gpu_train_batch_size      Batch size per GPU for training
    --gradient_accumulation_steps   Number of updates steps to accumulate before performing a backward/update pass
    --learning_rate                 The initial learning rate
    --weight_decay                  Weight decay
    --adam_betas                    betas for Adam optimizer
    --adam_epsilon                  Epsilon for Adam optimizer
    --no_clip_grad_norm             whether not to clip grad norm
    --max_grad_norm                 Max gradient norm
    --num_train_epochs              Total number of training epochs to perform
    --warmup_steps                  Linear warmup over warmup_steps
    --warmup_proportion             Linear warmup over warmup ratios
    --logging_steps                 Log every X updates steps
    --save_steps                    Save checkpoint every X updates steps   
```