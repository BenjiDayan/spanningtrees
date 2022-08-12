from datetime import timedelta
from typing import List, Union

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("gpt2")

import sys
sys.path.append('../../../AFLT/aflt-f2022')

from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize
import nltk
nltk.download('punkt')
SSP = SyllableTokenizer()

import string
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
cmu = cmudict.dict()

import tqdm

import time
import re
import pronouncing

from transformers import LogitsProcessor
import numpy as np
from string import ascii_lowercase, ascii_uppercase

import transformers

from transformers import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
    MaxLengthCriteria
)
import torch

from Phyme import Phyme

ph = Phyme()



tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")

import pickle
with open('vocab.pkl', 'rb') as file:
  vocab = pickle.load(file)



def set_scores_to_inf_for_banned_tokens(scores, banned_tokens):
    """
    Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
    list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

    Args:
        scores: logits distribution of shape (batch size, vocabulary size)
        banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return scores

    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))

    banned_mask = (
        torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    )
    scores = scores.masked_fill(banned_mask, -float("inf"))
    return scores



def get_text_syllables(text):
    words = word_tokenize(text)
    return [(word, count_syllables_cmu(word)) for word in words]

def get_text_num_syllables(text):
    return sum([x[1] for x in get_text_syllables(text)])


def count_syllables_sonhier(word):
    return len(SSP.tokenize(word))


def count_syllables_cmu(word):
    lower_word = word.lower()
    if lower_word in cmu:
        return max([len([y for y in x if y[-1] in string.digits])
                    for x in cmu[lower_word]])
    else:  # word isn't in cmu - hopefully is just punctuation
        # if re.search('[a-zA-Z]', lower_word):
        #     return count_syllables_sonhier(word)
        return 0


def poemify(text, syllables):
    """text: a str. syllables: a list of integers or just one integer."""
    new_text = ""
    syll_current = 0
    words = word_tokenize(text)
    syllables = [syllables] if type(syllables) is not list else syllables
    syllables = syllables * len(words)  # make sure to repeat more than enough lines
    syllables_iterator = iter(syllables)
    syll_target = next(syllables_iterator)
    for word in words:
        word_syll = count_syllables_cmu(word)
        if word_syll > 0:
            if syll_current == syll_target:
                syll_current = word_syll
                new_text += '\n' + word
                syll_target = next(syllables_iterator)
            elif syll_current + word_syll > syll_target:
                syll_current = syll_current + word_syll - syll_target
                new_text += word + '\n'  # This shouldn't really happen.
                syll_target = next(syllables_iterator)
            else:
                syll_current += word_syll
                new_text += ' ' + word
        else:
            new_text += ' ' + word

    return new_text


def quick_neatify_text(text):
  tokens = word_tokenize(text)
  word_syllables = [(x, count_syllables_cmu(x)) for x in tokens]
  return word_syllables


def get_last_word_indicators(word_syllables: List[int], rep: Union[int, List[int]]):
    """
    :param word_syllables: e.g. [2, 0, 4, 4, 1, 0, 1, 1, 5, 0, 1] - syllables for each word in the text
    :param rep: [7] for all lines 7 syllables, or [3,5] for alternating 3, 5
    :return current_line_num_syllables, line_num, last_word_indicator, syllable_target:

    """
    # rep: e.g. [7] for all lines 7 syllables, or [3,5] for alternating 3, 5
    # syllable lines.
    # e.g. [2, 0, 4, 4, 1, 0, 1, 1, 5, 0, 1] as input gives
    # [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]. so 3rd word is the last of
    # its line and so on.

    if type(rep) is int:
        rep = [rep]

    i = 0
    line_num = 0
    current_line_num_syllables = 0
    last_word_indicator = []
    while i < len(word_syllables):
        syllable_target = rep[line_num % len(rep)]
        current_line_num_syllables += word_syllables[i]
        if current_line_num_syllables >= syllable_target:
            line_num += 1
            current_line_num_syllables -= syllable_target
        last_word_indicator.append(line_num)
        i += 1

    # e.g. 0, 0, 1, 1, ... means the 3rd word is the first to add up to n_syllables. I.e. it's the last word of the line

    # 0, 0, 1, 1, .... - 0, 0, 0, 1, 1, ... = 0, 0, 1, 0, 0, ... indicators where that last word is. Each 1 is the last
    # word of the line (potentially followed by some punctuation/ other zero syllable words still on the same line which
    # don't count.
    last_word_indicator = np.concatenate([last_word_indicator, np.zeros(1)]) - np.concatenate(
        [np.zeros(1), last_word_indicator])

    return current_line_num_syllables, line_num, last_word_indicator, syllable_target


def get_last_syllable_breakpoint(word_syllables, n_syllables):
    word_num_syllables = [syllables for word, syllables in word_syllables]
    current_line_num_syllables, actual_line_num, last_word_indicator = get_last_word_indicators(word_syllables, n_syllables)

    last_word_index = np.where(last_word_indicator == 1)[0]
    if len(last_word_index) > 0:
      last_word_index = last_word_index[-1]
    else:
      last_word_index = None
    return last_word_index


syllable_words = []
idx_to_num_syllables = []
for i in range(15):
  syllable_words.append([])
for word in tqdm.tqdm(vocab['idx2sym'], total=len(vocab['idx2sym'])):
  ns = count_syllables_cmu(word)
  syllable_words[ns].append(word)
  idx_to_num_syllables.append(ns)

idx_to_num_syllables = np.array(idx_to_num_syllables)

class LongWordEncourager(LogitsProcessor):
    def __init__(self, num_syllables=3, sd=2):
        self.num_syllables = num_syllables
        self.idxs_of_num_syllables = []
        for num_syllables in range(15):
            self.idxs_of_num_syllables.append(np.where(idx_to_num_syllables == num_syllables)[0])

        self.idxs_of_high_num_syllables = np.where(idx_to_num_syllables >= self.num_syllables)[0]
        self.sd = sd  # "standard deviation of raw scores" - how much to add onto the longer words


    def __call__(self, input_ids, scores):
        scores[:, self.idxs_of_high_num_syllables] += self.sd
        return scores



def get_rhyming_words(word: str) -> List[str]:
    word = re.sub('[^a-z\']', '', word.lower())
    try:
        syllable_to_words = ph.get_perfect_rhymes(word, num_syllables=1)  # most permissive - only rhyme last syllable
    except KeyError:
        return []
    output = []
    for word_list in syllable_to_words.values():
        output += word_list
    return output



class BanNonWords(LogitsProcessor):
    def __init__(self):
        # We will allow the first three: <eos> , . Actually won't allow <eos> as
        # my poems get too short
        self.vocab_invalid_idx = [0] + [idx for idx, sym in enumerate(vocab['idx2sym']) if not sym in cmu][3:]

    def __call__(self, input_ids, scores):
        banned_tokens = []
        # for every beam (partially generated sentence)
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            banned_tokens.append(self.vocab_invalid_idx)
        # set the scores of all banned tokens over the beams to -inf
        scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores

class LineCommaPolice(LogitsProcessor):
    def __init__(self, num_syllables):
        self.num_syllables = num_syllables

    def __call__(self, input_ids, scores):
      # for every beam (partially generated sentence)
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            # get the last token of this beam
            text = tokenizer.decode(beam_input_ids)
            word_syllables = get_text_syllables(text)

            # how many syllables we're into the current line.
            current_line_num_syllables, actual_line_num, last_word_indicator, syllable_target = \
                get_last_word_indicators([sylls for word, sylls in word_syllables], self.num_syllables)

            if current_line_num_syllables == 0 and text[-1] != ',':
                beam_scores[2] += 20.0
        return scores


class LineRhymer(LogitsProcessor):
    def __init__(self, num_syllables=[3,5], sd=3.5):
        self.num_syllables = num_syllables
        self.idxs_of_num_syllables = []
        for num_syllables in range(15):
            self.idxs_of_num_syllables.append(np.where(idx_to_num_syllables == num_syllables)[0])

        # self.idxs_of_high_num_syllables = np.where(idx_to_num_syllables >= self.num_syllables)[0]
        # self.sd = sd  # "standard deviation of raw scores" - how much to add onto the longer words

        self.line_sep_id = vocab['sym2idx'][',']  # = 1

        cs_cmu = lambda x: count_syllables_cmu(x)
        cs_cmu = np.vectorize(cs_cmu)

        self.syllable_values = []
        self.syllable_idxs = []
        for i in range(1, 10):
            idxs = np.where(cs_cmu(vocab['idx2sym']) == i)
            self.syllable_idxs.append(idxs[0])
        self.sd = sd

    def __call__(self, input_ids, scores):

        banned_tokens = []
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            text = tokenizer.decode(beam_input_ids)
            word_syllables = get_text_syllables(text)

            # how many syllables we're into the current line.
            current_line_num_syllables, actual_line_num, last_word_indicator, syllable_target = \
                get_last_word_indicators([sylls for word, sylls in word_syllables], self.num_syllables)


            last_rhyme_word_index = np.where(last_word_indicator == 1)[0]
            if len(last_rhyme_word_index) > 0:
                last_rhyme_word_index = last_rhyme_word_index[-1]
            else:
                last_rhyme_word_index = None

            if last_rhyme_word_index:
                last_rhyme_word = word_syllables[last_rhyme_word_index][0]
                rhyming_words = set(get_rhyming_words(last_rhyme_word))
                rhyming_words_idxs = np.array([vocab['sym2idx'][word] for word in rhyming_words if
                                               word in vocab['sym2idx']])
            else:
                rhyming_words = set([])
                rhyming_words_idxs = np.array([])

            num_syllables = current_line_num_syllables
            target_syllables = self.num_syllables[actual_line_num % len(self.num_syllables)]
            # we're at the end of a line - don't ban as all welcome: 0, 1, 2, etc. syllable words
            if num_syllables == target_syllables:
                num_syllables = 0
                target_syllables = self.num_syllables[(actual_line_num+1) % len(self.num_syllables)]

            to_ban = [idxs for i, idxs in enumerate(self.syllable_idxs) if i + 1 + num_syllables > target_syllables]

            just_fit_idxs = self.syllable_idxs[target_syllables - num_syllables - 1]
            just_fit_rhyming_intersection = np.intersect1d(rhyming_words_idxs, just_fit_idxs, assume_unique=True)
            # just_fit_rhyming_intersection = set(just_fit_rhyming_intersection)  # indices within just_fit_idxs
            just_fit_idxs_ban = np.array([idx for idx in just_fit_idxs if not idx in just_fit_rhyming_intersection])

            beam_scores[just_fit_rhyming_intersection] += self.sd

            to_ban.append(just_fit_idxs_ban)

            if to_ban:
                print(f'num_syllables: {num_syllables}')
                print(text)
                print(quick_neatify_text(text))
                print(len(to_ban))
            banned_tokens.append(np.concatenate(to_ban) if to_ban else [])

        scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores


if __name__ == '__main__':
    def poemify(text, syllables):
        """text: a str. syllables: a list of integers or just one integer."""
        new_text = ""
        syll_current = 0
        words = word_tokenize(text)
        syllables = [syllables] if type(syllables) is not list else syllables
        syllables = syllables * len(words)  # make sure to repeat more than enough lines
        syllables_iterator = iter(syllables)
        syll_target = next(syllables_iterator)
        for word in words:
            word_syll = count_syllables_cmu(word)
            if word_syll > 0:
                if syll_current == syll_target:
                    syll_current = word_syll
                    new_text += '\n' + word
                    syll_target = next(syllables_iterator)
                elif syll_current + word_syll > syll_target:
                    syll_current = syll_current + word_syll - syll_target
                    new_text += word + '\n'  # This shouldn't really happen.
                    syll_target = next(syllables_iterator)
                else:
                    syll_current += word_syll
                    new_text += ' ' + word
            else:
                new_text += ' ' + word

        return new_text
    text = 'Over hill, over dale, meeting trade football fans, mail, and local teams, including "Catalunya Catalunya", "Catalunya" (ale), formed the youth of the dale. possible collegiate footballers'

    poemed = poemify(text, [6,7,8])
    print(poemed)


    logits_processor = LogitsProcessorList([long_word_encourager()])

    prompt = 'The wind rushed towards the man who lifted the spear from'
    prompt_tokenized = tokenizer(prompt, return_tensors='pt')
    prompt_tokenized = prompt_tokenized['input_ids']
    outputs = model.generate(
        input_ids=prompt_tokenized,
        num_return_sequences=2,
        no_repeat_ngram_size=2,
        remove_invalid_values=True,
        logits_processor=logits_processor,
        max_length=20,
        do_sample=True,
        top_p=0.92,
        top_k=0
        # stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
    )