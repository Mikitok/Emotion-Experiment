class Config(object):
    vocab_size=15000
    max_grad_norm = 5
    init_scale = 0.05
    hidden_size = 300
    attention_size=300
    lr_decay = 0.95
    valid_portion=0.0
    batch_size=5    #一次的文章数目
    sentence_size=50
    word_size=30
    keep_prob = 0.5
    #0.05
    learning_rate = 0.01
    max_epoch =2
    max_max_epoch =4
    num_label=6
    attention_iteration=3
    random_initialize=False
    embedding_trainable=True
    l2_beta=0.0