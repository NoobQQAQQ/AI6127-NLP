# Word-level Language Modeling using Transformer

To run the code, you can simply use the following command:

```bash
python main.py          # Train the model with default settings
```

Besides, you can assign different values to arguments to customize your running.
The `main.py` script accepts the following arguments:

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
The code will output all the results to the screen, an example is shown below.

```bash
Training:  10%|█         | 1/10 [00:59<08:51, 59.09s/it]epoch 1: val_loss 6.978016789061943 | val_ppl 1072.7886896305743
Training:  20%|██        | 2/10 [02:02<08:12, 61.57s/it]epoch 2: val_loss 6.8139212682992305 | val_ppl 910.4338719957888
...
Training: 100%|██████████| 10/10 [14:11<00:00, 85.17s/it]epoch 10: val_loss 5.927613270430091 | val_ppl 375.25780524984793
test_loss 5.850596081420787 | test_ppl 347.4414221423105
```

During training, the code will output the loss and perplexity on validation set per epoch. After training is done, these metrics on test set will be printed.