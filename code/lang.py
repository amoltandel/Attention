import torch
from torch.autograd import Variable
import torch.nn.functional as F

class EmbeddingVector:
    def __init__(self, vocab, en_dim = 300, vi_dim = 300, use_gpu = False):
        self.embeddings = {}
        self.use_gpu = use_gpu
        self.vocab = vocab
        self.en_dimensions = en_dim
        self.vi_dimensions = vi_dim
        self.embeddings['EN'] = torch.nn.Embedding(
                vocab['EN'].word_count,
                en_dim
            )
        self.embeddings['VI'] = torch.nn.Embedding(
                vocab['VI'].word_count,
                vi_dim
            )

    def get_embeddings(self, inp, lang='EN'):
        if lang != 'EN' and lang != 'VI':
            raise Exception('Language not supported')
        words = inp.split(' ')
        lst = []
        for word in words:
            if word in self.vocab[lang].word_dict.keys():
                lst.append(self.vocab[lang].word_dict[word])
            else:
                lst.append(self.vocab[lang].word_dict['<unk>'])
        tensor = Variable(torch.LongTensor(lst))
        output = self.embeddings[lang](tensor)
        if self.use_gpu:
            return output.cuda()
        return output

class Vocab:
    def __init__(self, filename='', size_limit = 5000, use_limit = True):
        self.file_name = filename
        word_list = open(self.file_name).read().split('\n')
        self.word_dict = {}
        self.int_word_dict = {}
        if use_limit:
            self.word_count = min(size_limit, len(word_list))
        else:
            self.word_count = len(word_list)
        for i in range(self.word_count):
            self.word_dict[word_list[i]] = i
            self.int_word_dict[i] = word_list[i]
    def get_key(self, var):
        if var in self.word_dict.keys():
            return self.word_dict[var]
        else:
            return self.word_dict['<unk>']
    def get_word(self, var):
        if var in self.int_word_dict.keys():
            return self.int_word_dict[var]
        else:
            return '<unk>'
