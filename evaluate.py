import _pickle

import matplotlib.pyplot as plt

from models import *
from data import *


def validation(model_config_file=None,
               train_file=None,
               test_file=None,
               tag_list='first',
               tag_format='BIO'):
    if model_config_file is None:
        model_config = ModelConfig()
    else:
        model_config = load_model_config(model_config_file)

    # 读取数据
    file_path = os.path.dirname(__file__)
    train_file = train_file or os.path.join(file_path, '../resources/corpus/train_data')
    test_file = test_file or os.path.join(file_path, '../resources/corpus/test_data')
    train_set = DataSet.ner_data_set(train_file)
    test_set = DataSet.ner_data_set(test_file)

    # 构建 word, tag词典
    tag_list = tag_lists[tag_list]
    tag2id = dict()
    for t in tag_list:
        tag2id[t] = len(tag2id)
    id2tag = dict(zip(tag2id.values(), tag2id.keys()))

    if model_config.embedding_matrix_path:
        word2id, embedding_matrix = load_w2v(model_config.embedding_matrix_path)
        model_config.embedding_matrix = embedding_matrix
    else:
        word2id = build_vocab(train_set.sentences)
        with open(os.path.join(model_config.save_path, 'word2id.cpkt'), 'wb') as f:
            _pickle.dump(word2id, f)
        model_config.vocab_size = len(word2id)
        save_config(model_config, model_config_file)

    processed_train_set = process_data_set(train_set, word2id, tag2id)

    # 训练模型
    model = make_model(model_config)
    model.fit(processed_train_set)

    # 模型预测
    _give_metric(model, test_set, word2id, tag2id, id2tag, tag_format)


def test(config_file,
         test_file=None,
         tag_list='first',
         tag_format='BIO',
         compare=False,
         config_file2=None,
         checkpoint_path1=None,
         checkpoint_path2=None):

    tag_list = tag_lists[tag_list]
    tag2id = dict()
    for t in tag_list:
        tag2id[t] = len(tag2id)
    id2tag = dict(zip(tag2id.values(), tag2id.keys()))

    if test_file is None:
        test_file = os.path.join(os.path.dirname(__file__), '../resources/corpus/test_data')
    test_set = DataSet.ner_data_set(test_file)

    if compare:
        assert config_file2 is not None
        assert checkpoint_path1 is not None
        assert checkpoint_path2 is not None
        config1 = load_model_config(config_file)
        config2 = load_model_config(config_file2)

        if config1.embedding_matrix_path:
            word2id, embedding_matrix = load_w2v(config1.embedding_matrix_path)
            config1.embedding_matrix = embedding_matrix
            config2.embedding_matrix = embedding_matrix
        else:
            with open(os.path.join(config1.save_path, 'word2id.cpkt'), 'wb') as f:
                word2id = _pickle.load(f)

        g1 = tf.Graph()
        g2 = tf.Graph()

        with g1.as_default():
            model1 = make_model(config1)
            model1.restore(os.path.join(config1.save_path, checkpoint_path1))
        with g2.as_default():
            model2 = make_model(config2)
            model2.restore(os.path.join(config2.save_path, checkpoint_path2))

        _, _, f1 = _give_metric(model1, test_set, word2id, tag2id, id2tag, tag_format, mode='all')
        _, _, f2 = _give_metric(model2, test_set, word2id, tag2id, id2tag, tag_format, mode='all')
        plt.plot(f1, f2, '.')
        plt.plot([0, 1], [0, 1], 'r-')
        plt.show()
    else:
        config = load_model_config(config_file)

        if config.embedding_matrix_path:
            word2id, embedding_matrix = load_w2v(config.embedding_matrix_path)
            config.embedding_matrix = embedding_matrix
        else:
            with open(os.path.join(config.save_path, 'word2id.cpkt'), 'wb') as f:
                word2id = _pickle.load(f)

        model = make_model(config)

        with open('./model_save/result.csv', 'a', encoding='utf-8') as rf:
            rf.write(config_file + "\n")

        if checkpoint_path1 is not None:
            model.restore(os.path.join(config.save_path, checkpoint_path1))
            p, r, f = _give_metric(model, test_set, word2id, tag2id, id2tag, tag_format)
            with open('./model_save/result.csv', 'a', encoding='utf-8') as rf:
                rf.write('{}\t{}\t{}\n'.format(p, r, f))
        else:
            checkpoint_paths = tf.train.get_checkpoint_state(config.save_path).all_model_checkpoint_paths

            for i, path in enumerate(checkpoint_paths):
                model.restore(path)
                p, r, f = _give_metric(model, test_set, word2id, tag2id, id2tag, tag_format)
                with open('./model_save/result.csv', 'a', encoding='utf-8') as rf:
                    rf.write('{}\t{}\t{}\n'.format(p, r, f))


def _give_metric(model, test_set, word2id, tag2id, id2tag, tag_format, mode="mean"):
    processed_test_set = process_data_set(test_set, word2id, tag2id)
    preds = []
    sentences = np.expand_dims(processed_test_set.sentences, 1)
    lengths = np.expand_dims(processed_test_set.lengths, 1)

    class TmpSet(object):
        pass

    for s, l in zip(sentences, lengths):
        tmp_set = TmpSet()
        tmp_set.sentences = s
        tmp_set.lengths = l
        pred = model.predict(tmp_set)
        preds.append(pred)

    pred = np.vstack(preds)
    tags = ids_to_tags(pred, id2tag, test_set.lengths)
    prediction = split_by_tags(test_set.sentences, tags, tag_format)
    gold = split_by_tags(test_set.sentences, test_set.tags, tag_format)

    # 给出p, r, f1指标
    p, r, f1 = evaluate_all(prediction, gold, mode)
    print("precision: {}\nrecall: {}\nf1 score: {}".format(p, r, f1))
    return p, r, f1


def precision(prediction, gold):
    """针对一句话的精度评估

    :param prediction: 模型的预测结果
    :param gold: 测试数据的标注
    :return: precision 精度
    """
    p, _, _ = evaluate(prediction, gold)
    return p


def recall(prediction, gold):
    """针对一句话的召回率评估

    :param prediction: 模型的预测结果
    :param gold: 测试数据的标注
    :return: recall 召回率
    """
    _, r, _ = evaluate(prediction, gold)
    return r


def f1_score(prediction, gold):
    """

    :param prediction: 模型的预测结果
    :param gold: 测试数据的标注
    :return: recall 召回率
    """
    _, _, f1 = evaluate(prediction, gold)

    return f1


def evaluate(prediction, gold):
    """针对一句话的指标评估

    :param prediction: 模型的预测结果
    :param gold: 测试数据的标注
    :return: (p, r, f1), 精度，召回率， F1值
    """

    right = 0
    gold_num = len(gold)
    detection_num = len(prediction)

    if gold_num == 0 or detection_num == 0:
        return 0, 0, 0

    for p in prediction:
        word, tag, start = p['word'], p['tag'], p['start']
        for i in range(right, gold_num):
            w_true, t_true, s_true = gold[i]['word'], gold[i]['tag'], gold[i]['start']
            if word == w_true and tag == t_true and start == s_true:
                right += 1
                break

    if right == 0:
        return 0, 0, 0

    p = right / detection_num
    r = right / gold_num
    f1 = 2 * (p * r) / (p + r)
    return p, r, f1


def evaluate_all(prediction, gold, mode='all'):
    """

    :param prediction: 整个测试数据集的预测结果
    :param gold: 整个测试数据集的标注
    :param mode: 'mean', 'all'. mean表示对最后的结果做加权平均， all表示以列表形式输出每个句子各自的p,r,f指标
    :return:
    """
    p, r, f = [], [], []
    detect_num = []
    gold_num = []
    for pred, g in zip(prediction, gold):
        metric = evaluate(pred, g)
        p.append(metric[0])
        r.append(metric[1])
        f.append(metric[2])
        detect_num.append(len(pred))
        gold_num.append(len(g))

    if mode == 'all':
        return p, r, f
    else:
        p = sum(map(lambda z: z[0] * z[1], zip(p, detect_num))) / sum(detect_num)
        r = sum(map(lambda z: z[0] * z[1], zip(r, gold_num))) / sum(gold_num)
        f = 2 * (p * r) / (p + r)
        return p, r, f
