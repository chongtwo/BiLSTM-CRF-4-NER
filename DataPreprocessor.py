import codecs
import os


class DataPreprocessor(object):

    # 将连续的文字转换为一行一个
    def character_tagging(self, input_file, output_file):
        input_data = codecs.open(input_file, 'r', 'utf-8')
        output_data = codecs.open(output_file, 'w', 'utf-8')
        for line in input_data.readlines():
            word_with_tag_list = line.strip().split()
            for word_with_tag in word_with_tag_list:
                (word, tag) = word_with_tag.split("/")
                if tag == "o":
                    for w in word[0:len(word_with_tag)]:
                        output_data.write(w + "\t" + "O" + "\n")
                else:
                    if len(word_with_tag) == 1:
                        output_data.write(word + "\tB-" + tag + "\n")
                    else:
                        output_data.write(word[0] + "\tB-" + tag + "\n")
                        for w in word[1:len(word) - 1]:
                            output_data.write(w + "\tI-" + tag + "\n")
            output_data.write("\n")
        input_data.close()
        output_data.close()

    def make_word_level_w2v_train_data(self, input_file, output_file):
        input_data = codecs.open(input_file, 'r', 'utf-8')
        output_data = codecs.open(output_file, 'w', 'utf-8')
        for line in input_data.readlines():
            word_with_tag_list = line.strip().split()
            for word_with_tag in word_with_tag_list:
                (word, tag) = word_with_tag.split("/")
                output_data.write(word+" ")
            output_data.write("\n")
        input_data.close()
        output_data.close()

    def make_char_level_w2v_train_data(self, input_file, output_file):
        input_data = codecs.open(input_file, 'r', 'utf-8')
        output_data = codecs.open(output_file, 'w', 'utf-8')
        for line in input_data.readlines():
            word_with_tag_list = line.strip().split()
            for word_with_tag in word_with_tag_list:
                (word, tag) = word_with_tag.split("/")
                for char in word:
                    output_data.write(char + " ")
            output_data.write("\n")
        input_data.close()
        output_data.close()

if __name__ == '__main__':
    data_preprocessor = DataPreprocessor()
    inputFile = os.path.join('.', "data", 'train1.txt')
    # outputFile_rnn = os.path.join('.', 'data', "train")
    # data_preprocessor.character_tagging(inputFile, outputFile_rnn)
    outputFile_w2v = os.path.join('.', 'data', "w2v_raw_char")
    data_preprocessor.make_char_level_w2v_train_data(inputFile, outputFile_w2v)
