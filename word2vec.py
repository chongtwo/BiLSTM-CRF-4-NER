# -*- coding: utf-8 -*-
import logging
from gensim.models import word2vec
# from gensim.models.word2vec import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import jieba
import jieba.analyse
import os


class MyWord2Vec(object):

    def gensim_example(self):
        # path = get_tmpfile("word2vec.model")
        # print(path);

        # 训练并保存
        # model = Word2Vec(common_texts, size=100, window=5, min_count=1)
        # model.save("word2vec.model")

        model = word2vec.Word2Vec.load("word2vec.model")
        # model.train([["hello","world"]], total_examples=1, epochs=1)
        vector = model.wv['computer']

        print("gensim_example")
        print(vector)

    def jieba_seg(self, raw_path, seg_path, user_dict):
        for word in user_dict:
            jieba.suggest_freq(word, True)
        with open(raw_path, encoding='UTF-8') as f:
            document = f.read()
            document_cut = jieba.cut(document)
            result = ' '.join(document_cut)
            with open(seg_path, 'w', encoding="UTF-8") as f2:
                f2.write(result)

        print("jieba_seg")

    # 找出某一个词向量最相近的词集合
    def get_similar_word(self, model, word):
        for key in model.wv.similar_by_word(word, topn=5):
            print(key[0],key[1])



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    w2v = MyWord2Vec()
    # rawPath = './in_the_name_of_people.txt'
    # segPath = './in_the_name_of_people_segment.txt'
    # userDict =['沙瑞金','田国富','高育良''侯亮平','钟小艾','陈岩石', '欧阳菁','易学习','王大路',
    #            '蔡成功','孙连城','季昌明','丁义珍','郑西坡', '赵东来''高小琴','赵瑞龙', '林华华',
    #           '陆亦可', '刘新建', '刘庆祝']
    # w2v.jieba_seg(rawPath, segPath, userDict)
    train_data_path = os.path.join('.', 'data', "w2v_raw_char")
    sentences = word2vec.LineSentence(train_data_path)
    char_model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)
    char_model.save("char_model.model")
    # model = Word2Vec.load("word2vec.model")
    w2v.get_similar_word(char_model, "我")



