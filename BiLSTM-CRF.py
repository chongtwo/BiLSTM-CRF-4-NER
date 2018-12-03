import tensorflow as tf
from gensim.models import word2vec, KeyedVectors
import codecs
import os
import numpy as np


# load model
char_model = word2vec.Word2Vec.load("char_model.model")
# char_embeddings = KeyedVectors.load("char_model.model", mmap='r')
word_vector_size = char_model.wv.vector_size
vocab_size = len(char_model.wv.vocab)

# 创建 序号-向量 数组
char_embedding = np.zeros(shape=(vocab_size, word_vector_size), dtype='int32')
char2id_dict = {}
for idx, word in enumerate(sorted(char_model.wv.vocab)):
    char_embedding[idx] = char_model.wv.get_vector(word)
    char2id_dict[word] = idx


# 将 序号-向量 数组转化为tensor
with tf.variable_scope("input_layer"):
    embedding_tf = tf.get_variable("embedding", [vocab_size, word_vector_size],
                                   initializer=tf.constant_initializer(char_embedding),
                                   trainable=False)

#  根据输入文字，创建输入矩阵
batch_size = 100
sentence_max_len = 50
input_file = os.path.join('.','data','w2v_raw_char')
input_data = codecs.open(input_file, 'r', 'utf-8').readlines()
for total_line_index in range(0, len(input_data) - 1, batch_size):
    batch_data = input_data[total_line_index: total_line_index + batch_size - 1]
    word_ids = tf.placeholder(tf.int32, shape=[batch_size, sentence_max_len])
    word_ids_list = np.zeros(shape=(batch_size, sentence_max_len))
    for line_index, line in enumerate(input_data):
        line = line.split()
        print(line_index)
        for char_index, char in enumerate(line):
            char_id = char2id_dict[char]
            word_ids_list[line_index][char_index] += char_id

sequence_lengths = tf.placeholder(tf.int32, shape=[batch_size])
# L = tf.Variable(char_model, trainable=False)

# 一个批次的数据 shape = (batch, sentence, word_vector_size)
pretrained_embeddings = tf.nn.embedding_lookup(embedding_tf, word_ids)

# contextual training
hidden_size = 100
num_tags = 7

with tf.Session() as sess:
    print(sess.run(pretrained_embeddings, feed_dict={word_ids: word_ids_list}))

cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)
(outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, pretrained_embeddings , sequence_length=sequence_lengths)
# outputs是个二元组
# 将两个LSTM的输出合并
output_fw, output_bw = outputs
output = tf.concat([output_fw, output_bw], axis=-1)

# 变换矩阵，可训练参数
W = tf.get_variable("W", [2 * hidden_size, num_tags])

# 线性变换
matricized_output = tf.reshape(output, [-1, 2 * hidden_size])
matricized_unary_scores = tf.matmul(matricized_output, W)
unary_scores = tf.reshape(matricized_unary_scores, [batch_size, max_seq_len, num_tags])

# Loss函数
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, tags, sequence_lengths)
loss = tf.reduce_mean(-log_likelihood)

