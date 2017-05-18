# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:40:28 2017

@author: lyy
"""

import numpy as np, pandas as pd, re
from collections import Counter
from os import chdir
chdir(r'F:\workspace\python\project\spyder3\course\deep-learning\tv-script-generation')

text = open('data/simpsons/moes_tavern_lines.txt').read()[81:]


print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

#replace token
token_dict = {'.':'<PERIOD>', ',':'<COMMA>', '"':'<QUOTATION_MARK>', ';':'<SEMICOLON>', '!':'<EXCLAMATION_MARK>', \
'?':'<QUESTION_MARK>', '(':'<LEFT_PAREN>', ')':'<RIGHT_PAREN>', '--':'<dash>', '\n':'<return>'}

s_dict = {'.':'',',':'','"':'',';':'','!':'','?':'','(':'',')':'','--':'','&':'',
'$':'','%':''}
for key,value in s_dict.items():
    text = text.replace(key,value)
    
from string import punctuation
all_text = ''.join([c for c in text if c not in punctuation])
#all_text = all_text.replae('\n',' ')
text = all_text.split('\n')

all_text = ' '.join(text)
words = all_text.split() 

words=[re.sub('\d','',value) for row in text.split('\n') for value in row.split(' ')]
word_counts = Counter(words)
sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

from collections import Counter

def create_lookup_tables(text):
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return (vocab_to_int, int_to_vocab)

#build rnn
Outputs, FinalState = tf.nn.dynamic_rnn(cell, inputs, sequence_length=[50],
                     dtype = tf.float32, parallel_iterations=None,
                     swap_memory=False, time_major=False, scope=None)