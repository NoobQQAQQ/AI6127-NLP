To run the code, you can simply use the following commands:

```bash
bash BASH_FILE DATA_DIR TASK_NAME     # Train the model with default settings
bash run_albert.sh DATA_DIR TASK_NAME   # Train the model with default settings
bash run_roberta.sh DATA_DIR TASK_NAME  # Train the model with default settings
bash run_xlnet.sh DATA_DIR TASK_NAME    # Train the model with default settings
```



Besides, you can assign different values to arguments to customize your running.
The script accepts the following arguments:

```bash
optional arguments:
  --data_path ../data   location of the data corpus
  --emb_dim 512         size of word embeddings
  --nhead 8             the number of heads in the encoder of the transformer model
  --N 2                 number of layers
  --ffn_dim 256         size of hidden layers
  --lr 20               initial learning rate
  --epochs 10           upper epoch limit
  --train_bs 128        batch size for training
  --eval_bs 128         batch size for testing
  --seq_len 35          sequence length
  --dropout 0.2         dropout probability applied to layers
  --no_scale            no using scaling in the attention head
```

With these arguments, a variety of models can be tested.

During training, the code will output the loss and perplexity on validation set per epoch.
After training is done, these metrics on test set will be printed.