class Args:
    train_path = "./data/train_process.json"
    test_path = "./data/test_process.json"
    seq_labels_path = "./data/intents.txt"
    token_labels_path = "./data/slots.txt"
    bert_dir = "../../ProjectsNLP/models/model-hub/chinese_wwm_ext_pytorch/"
    save_dir = "./checkpoints/"
    load_dir = "./checkpoints/model_20230614190301.pt"
    do_train = False
    do_eval = False
    do_test = False
    do_save = False
    do_predict = True
    load_model = True
    device = None
    seqlabel2id = {}
    id2seqlabel = {}
    with open(seq_labels_path, "r") as fp:
        seq_labels = fp.read().split("\n")
        for i, label in enumerate(seq_labels):
            seqlabel2id[label] = i
            id2seqlabel[i] = label

    tokenlabel2id = {}
    id2tokenlabel = {}
    with open(token_labels_path, "r") as fp:
        token_labels = fp.read().split("\n")
        for i, label in enumerate(token_labels):
            tokenlabel2id[label] = i
            id2tokenlabel[i] = label

    tmp = ["O"]
    for label in token_labels:
        B_label = "B-" + label
        I_label = "I-" + label
        tmp.append(B_label)
        tmp.append(I_label)
    nerlabel2id = {}
    id2nerlabel = {}
    for i, label in enumerate(tmp):
        nerlabel2id[label] = i
        id2nerlabel[i] = label

    hidden_size = 768
    seq_num_labels = len(seq_labels)
    token_num_labels = len(tmp)
    max_len = 34
    batchsize = 64
    lr = 2e-5
    epoch = 10
    hidden_dropout_prob = 0.1


if __name__ == "__main__":
    args = Args()
    print(args.seq_labels)
    print(args.seqlabel2id)
    print(args.tokenlabel2id)
    print(args.nerlabel2id)
