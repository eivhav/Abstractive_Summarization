
from __future__ import unicode_literals, print_function, division
import json
import glob


from io import open
import unicodedata
import random
import operator
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3






class Vocab:
    def __init__(self, limit):
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK", 4: "HIGH-abs", 5:"LOW-abs", 6: "NEXT-line"}
        self.word2index = {self.index2word[k]: k for k in self.index2word}
        self.word2count = {}
        self.n_words = len(self.index2word)  # Count SOS and EOS
        self.vocab_size = 0
        self.vocab_incl_size = len(self.index2word) + limit



class TextPair:
    def __init__(self, source, target, id, vocab,
                 max_target_n=5, novelty_vec=None, novelty_deg=None, ctrl_method='vector_ngram', next_lines=False):
        self.id = id
        self.source = source
        self.target = target
        self.full_source_tokens = []
        self.masked_source_tokens = []
        self.full_target_tokens = []
        self.masked_target_tokens = []
        self.unknown_tokens = dict()
        self.tri_gram_novelty_vector = []
        self.tri_gram_novelty_degree = 1
        self.source_tri_grams = dict()

        self.add_sentence(source, (self.full_source_tokens, self.masked_source_tokens), vocab)
        s = self.full_source_tokens
        for t in range(len(source[2:400])):
            self.source_tri_grams[str(s[t]) + "~" + str(s[t+1]) + "~" + str(s[t+2])] = True

        for t in target:
            self.add_sentence(t, (self.full_target_tokens, self.masked_target_tokens), vocab, last=False)
        self.full_target_tokens.append(EOS_token)
        self.masked_target_tokens.append(EOS_token)


    def get_text(self, tokens, vocab):
        words = []
        for t in tokens:
            if t in vocab.index2word: words.append(vocab.index2word[t])
            else: [words.append(w) for w in self.unknown_tokens if self.unknown_tokens[w] == t]
        return " ".join(words)

    def get_tokens(self, words, vocab):
        tokens = []
        for w in words:
            if w in self.unknown_tokens: tokens.append(self.unknown_tokens[w])
            elif w in vocab.word2index: tokens.append(vocab.word2index[w])
            else:
                tokens.append(4)
        return tokens

    def add_sentence(self, sentence, destination, vocab, last=True):
        if isinstance(sentence, str): sentence = sentence.split(" ")
        for w in sentence:
            try:
                if w in vocab.word2index:
                    destination[0].append(vocab.word2index[w])
                    destination[1].append(vocab.word2index[w])
                else:
                    if w not in self.unknown_tokens:
                        self.unknown_tokens[w] = len(vocab.index2word) + len(self.unknown_tokens)
                    destination[0].append(self.unknown_tokens[w])
                    destination[1].append(UNK_token)
            except:
                print("FAILED FOR WORD", w)
                continue
        if last:
            destination[0].append(EOS_token)
            destination[1].append(EOS_token)


    def compute_novelty(self, tokens):
        tri_grams = [str(tokens[i]) + "~" + str(tokens[i+1]) + "~" + str(tokens[i+2]) for i in range(len(tokens)-2)]
        if len(tri_grams) == 0: return 0
        return sum([1 for tri in tri_grams if tri not in self.source_tri_grams]) / len(tri_grams)



