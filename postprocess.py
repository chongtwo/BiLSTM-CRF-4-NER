

def ids_to_tags(ids, id2tag, lengths=None):
    if lengths is None:
        return list(map(lambda x: [id2tag.get(i) for i in x], ids))
    else:
        tags = []
        for id_, length in zip(ids, lengths):
            tags.append([id2tag.get(i) for i in id_[-length:]])
        return tags


def split_by_tags(sentences, tags, tag_format='BIO'):
    """

    :param list sentences: sentences [['今', '日', '查', '房'], ['患', '者'], ....] or ['今日查房', '患者....', ...]
    :param list tags: [['B', 'I', 'O'], ['B', 'I', 'O'], ...]
    :param str tag_format: 'BIO' or 'BMESO'
    :return:
    """
    result = []
    if tag_format not in ['BIO', 'BMESO']:
        raise ValueError('unsupported tag format')
    for sentence, tag in zip(sentences, tags):
        one_result = []
        if isinstance(sentence, list):
            sentence = ''.join(sentence)
        if tag_format == 'BIO':
            start, end = 0, 0
            for i in range(len(tag)):
                if tag[i][0] == 'B':
                    start = i
                elif tag[i][0] == 'I':
                    if i != len(tag) - 1:
                        if tag[i+1][0] == 'B':
                            end = i + 1
                            one_result.append({'word': sentence[start:end],
                                               'start': start, 'end': end, 'tag': tag[start][2:]})
                elif tag[i][0] == 'O':
                    if i == 0:
                        continue
                    else:
                        if tag[i-1][0] == 'B' or tag[i-1][0] == 'I':
                            end = i
                            one_result.append({'word': sentence[start:end],
                                               'start': start, 'end': end, 'tag': tag[start][2:]})
                if i == len(tag)-1 and end < start:
                    end = i + 1
                    one_result.append({'word': sentence[start:end],
                                       'start': start, 'end': end, 'tag': tag[start][2:]})
        else:
            start, end = 0, 0
            for i, t in enumerate(tag):
                if t[0] == 'B':
                    start = i
                elif t[0] == 'E':
                    end = i + 1
                    one_result.append({'word': sentence[start:end], 'start': start, 'end': end, 'tag': tag[start][2:]})
                elif t[0] == 'S':
                    start = i
                    end = i + 1
                    one_result.append({'word': sentence[start:end], 'start': start, 'end': end, 'tag': tag[start][2:]})

        result.append(one_result)
    return result
