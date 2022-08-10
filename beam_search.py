from datetime import timedelta

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("gpt2")

import sys
sys.path.append('../../../AFLT/aflt-f2022')

from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State


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

import time
import re
import pronouncing

from transformers import LogitsProcessor
import numpy as np
from string import ascii_lowercase, ascii_uppercase


# You can visualize this FSA to convince yourself it works
def three_consecutive_word_FSA(letter_a=["a", "A"], letter_b=["b", "B"], letter_c=["c", "C"]):
    fsa = FSA(Boolean)
    space = " "
    special_chars = letter_a + letter_b + letter_c

    fsa.set_I(State(0))

    # Benji rough interpretation (not rigorous): This FSA basically reads in characters one at a time. Most states
    # have a self-arc accepting all characters but spaces. Spaces transition from
    # one state to another. An intial 'a' char transitions us to the "word begun with a" state
    # , which we can only get out of by taking a Space arc - we're now in a state which
    # has an arc to "word begun with a b" for char b, and an arc to "dead" state for any other
    # char. All states are final except this "dead" state. Additionally starting with a non "a" word
    # is ok, we keep going until we start a word with an a.

    for c in ascii_lowercase + ascii_uppercase + space:
        if c in letter_a:
            fsa.add_arc(State(0), Sym(c), State(1))
            fsa.add_arc(State(6), Sym(c), State(1))
            fsa.add_arc(State(7), Sym(c), State(7))
            fsa.add_arc(State(1), Sym(c), State(1))
            fsa.add_arc(State(3), Sym(c), State(3))
            fsa.add_arc(State(5), Sym(c), State(5))
            fsa.add_arc(State(2), Sym(c), State(8))
            fsa.add_arc(State(4), Sym(c), State(8))
        elif c in letter_b:
            fsa.add_arc(State(2), Sym(c), State(3))
            fsa.add_arc(State(0), Sym(c), State(7))
            fsa.add_arc(State(7), Sym(c), State(7))
            fsa.add_arc(State(1), Sym(c), State(1))
            fsa.add_arc(State(3), Sym(c), State(3))
            fsa.add_arc(State(5), Sym(c), State(5))
            fsa.add_arc(State(4), Sym(c), State(8))
            fsa.add_arc(State(6), Sym(c), State(8))
        elif c in letter_c:
            fsa.add_arc(State(4), Sym(c), State(5))
            fsa.add_arc(State(0), Sym(c), State(7))
            fsa.add_arc(State(7), Sym(c), State(7))
            fsa.add_arc(State(1), Sym(c), State(1))
            fsa.add_arc(State(3), Sym(c), State(3))
            fsa.add_arc(State(5), Sym(c), State(5))
            fsa.add_arc(State(2), Sym(c), State(8))
            fsa.add_arc(State(6), Sym(c), State(8))
        elif c == space:
            fsa.add_arc(State(1), Sym(c), State(2))
            fsa.add_arc(State(3), Sym(c), State(4))
            fsa.add_arc(State(5), Sym(c), State(6))
            fsa.add_arc(State(7), Sym(c), State(0))
            fsa.add_arc(State(0), Sym(c), State(7))
            fsa.add_arc(State(2), Sym(c), State(8))
            fsa.add_arc(State(4), Sym(c), State(8))
            fsa.add_arc(State(6), Sym(c), State(8))
        else:
            fsa.add_arc(State(0), Sym(c), State(7))
            fsa.add_arc(State(7), Sym(c), State(7))
            fsa.add_arc(State(1), Sym(c), State(1))
            fsa.add_arc(State(3), Sym(c), State(3))
            fsa.add_arc(State(5), Sym(c), State(5))
            fsa.add_arc(State(2), Sym(c), State(8))
            fsa.add_arc(State(4), Sym(c), State(8))
            fsa.add_arc(State(6), Sym(c), State(8))

    fsa.set_F(State(0))
    fsa.set_F(State(1))
    fsa.set_F(State(2))
    fsa.set_F(State(3))
    fsa.set_F(State(4))
    fsa.set_F(State(5))
    fsa.set_F(State(6))
    fsa.set_F(State(7))

    return fsa


def even_FSA():
    fsa = FSA(Boolean)

    fsa.set_I(State(0))
    for c in ascii_lowercase + ascii_uppercase:
        fsa.add_arc(State(0), Sym(c), State(1))
        fsa.add_arc(State(1), Sym(c), State(0))
    fsa.set_F(State(0))

    return fsa

# src: https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html

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


class WordLogits(LogitsProcessor):

    def __init__(self, fsa, vocab):
        self.fsa = fsa
        self.keys = np.array(list(vocab.keys()))  # strings for each word
        self.values = np.array(list(vocab.values()))  # integer ids for each word

        # add token to banned tokens if the fsa does not accept
        accepted = lambda x: self.fsa.accept(x.strip('Ġ '))
        vec_accepted = np.vectorize(accepted)
        indexes = np.where(vec_accepted(self.keys) == Boolean(0))[0]
        self.banned_values = self.values[indexes]

    def __call__(self, input_ids, scores):
        banned_tokens = []
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            banned_tokens.append(self.banned_values)

        # set the scores of all banned tokens over the beams to -inf
        scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores


class ConsecutiveLogits(LogitsProcessor):
    def __init__(self, fsa, vocab):
        self.fsa = fsa
        self.keys = np.array(list(vocab.keys()))
        self.values = np.array(list(vocab.values()))

    def __call__(self, input_ids, scores):
        banned_tokens = []
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            start = time.time()
            main_beam_string = tokenizer.decode(beam_input_ids)
            print(main_beam_string)

            # func which takes in our string so far, adds on our potential next word and runs it through
            # TODO did removing str help?
            accepted = lambda x: self.fsa.accept(main_beam_string + x.replace("Ġ", " "))
            vec_accepted = np.vectorize(accepted)
            indexes = np.where(vec_accepted(self.keys) == Boolean(0))[0]

            banned_tokens.append(self.values[indexes])

            end = time.time()
            elapsed = (end - start)
            print("Time: ", str(timedelta(seconds=elapsed)))
            start = end

        scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores


class ConsecutiveLogitsWithDictionary(LogitsProcessor):

    # Function for easy fsa arc adding
    def add_arcs_string(self, added_string, fsa, start_ind=0):

        for i, x in enumerate(list(added_string)):
            curr_i = i + start_ind
            fsa.add_arc(State(curr_i), Sym(x), State(curr_i + 1), Boolean(1))

    def __init__(self, fsa, vocab):

        self.constraint_fsa = fsa
        self.keys = np.array(list(vocab.keys()))
        self.values = np.array(list(vocab.values()))
        self.string_fsas = dict()

    def __call__(self, input_ids, scores):

        banned_tokens = []
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):

            start = time.time()

            main_beam_string = str(tokenizer.decode(beam_input_ids))
            print(main_beam_string)

            # create fsa until this point if not created
            if main_beam_string not in self.string_fsas:
                temp_fsa = FSA(R=Boolean)
                self.add_arcs_string(main_beam_string, temp_fsa)
                temp_fsa.set_I(State(0), Boolean(1))
                self.string_fsas[main_beam_string] = temp_fsa
            else:
                temp_fsa = self.string_fsas[main_beam_string]

            def check_banned(x):
                new_fsa = temp_fsa.copy()
                added = str(x.replace("Ġ", " "))
                new_string = str(main_beam_string + added)
                self.add_arcs_string(added_string=added, fsa=new_fsa, start_ind=len(main_beam_string))
                new_fsa.add_F(State(len(new_string)), Boolean(1))
                return self.constraint_fsa.intersect(new_fsa).pathsum()

            vec_accepted = np.vectorize(check_banned)
            indexes = np.where(vec_accepted(self.keys) == Boolean(0))[0]

            banned_tokens.append(self.values[indexes])

            end = time.time()
            elapsed = (end - start)
            print("Time: ", str(timedelta(seconds=elapsed)))
            start = end

        scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores


from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
    MaxLengthCriteria
)
import torch

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
    syll_target = syllables.pop()
    words = word_tokenize(text)
    for word in words:
        word_syll = count_syllables_cmu(word)
        if syll_current == syll_target and word_syll > 0:
            syll_current = word_syll
            new_text += '\n' + word
        elif syll_current < syll_target:
            syll_current += word_syll
            if syll_current > syll_target:
                raise Exception('word too many syllables')
            new_text += ' ' + word

    return new_text


def quick_neatify_text(text):
  tokens = word_tokenize(text)
  word_syllables = [(x, count_syllables_cmu(x)) for x in tokens]
  return word_syllables

def get_last_syllable_breakpoint(word_syllables, n_syllables):
    syllable_cumsum = np.cumsum([syllables for word, syllables in word_syllables])
    word_line_nums = syllable_cumsum // n_syllables
    last_word_indicator = np.concatenate([word_line_nums, np.zeros(1)]) - np.concatenate([np.zeros(1), word_line_nums])
    last_word_index = np.where(last_word_indicator == 1)[0]
    if len(last_word_index) > 0:
      last_word_index = last_word_index[-1]
    else:
      last_word_index = None
    return last_word_index


class long_word_encourager(LogitsProcessor):
    def __init__(self):
        # create an array of tokens
        # remove the 'Ġ' token (used to represent a blank space in the tokenizer)
        self.keys = list(tokenizer.vocab.keys())
        index_to_pop = self.keys.index('Ġ')
        self.keys.pop(index_to_pop)

        stripped = np.vectorize(lambda x: x.strip('Ġ '))
        # self.keys = np.array(stripped(self.keys))
        self.keys = np.array(self.keys)
        self.keys_stripped = stripped(self.keys)

        # create an array of ids
        # also remove the 'Ġ' token
        self.values = list(tokenizer.vocab.values())
        self.values.pop(index_to_pop)
        self.values = np.array(self.values)


        # dictionary mapping token value to key/value list index
        self.value_to_idx = {val: idx for idx, val in enumerate(self.values)}
        self.word_to_value = {word: self.values[idx] for idx, word in enumerate(self.keys)}

        cs_cmu = lambda x: count_syllables_cmu(x)
        cs_cmu = np.vectorize(cs_cmu)

        self.add_on_words = [x for x in self.keys if x[0] != 'Ġ']
        self.add_on_words_idxs = [i for i, x in enumerate(self.keys) if x[0] != 'Ġ']

        cs_cmu = lambda x: count_syllables_cmu(x)
        cs_cmu = np.vectorize(cs_cmu)
        self.syllable_values = []
        self.syllable_idxs = []
        for i in range(1, 10):
            idxs = np.where(cs_cmu(self.keys) == i)
            self.syllable_idxs.append(idxs[0])
            self.syllable_values.append(self.values[idxs])

    def __call__(self, input_ids, scores):
        print('hi')
        mean = torch.mean(scores)
        sd = torch.std(scores)
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            # text = tokenizer.decode(beam_input_ids)
            # word_syllables = get_text_syllables(text)
            # last_word_syllable_length = word_syllables[-1][-1]
            # if last_word_syllable_length

            for j, add_on_word in enumerate(self.add_on_words):
                tokens = [self.word_to_value[add_on_word]]
                i = -1
                while True:
                    token = beam_input_ids[i]
                    word = tokenizer.decode(token)
                    tokens.append(token)
                    if word[0] == ' ' and len(word) > 1:
                        break
                    i -= 1
                constructed = tokenizer.decode(tokens[::-1])
                constructed_previous = tokenizer.decode(tokens[1:][::-1])
                delta_syllables = get_text_num_syllables(constructed) - get_text_num_syllables(constructed_previous)
                bad = num_syllables - delta_syllables < 0 or num_syllables + delta_syllables > self.n_syllables
                if bad:
                    add_on_word_idxs_to_ban.append(self.add_on_words_idxs[i])
                if delta_syllables + num_syllables == self.n_syllables:
                    if not constructed in rhyming_words:
                        add_on_word_idxs_to_ban.append(self.add_on_words_idxs[i])
        return scores

class n_word_lines2(LogitsProcessor):
    def __init__(self, n_syllables=7):
        self.n_syllables = n_syllables

        # create an array of tokens
        # remove the 'Ġ' token (used to represent a blank space in the tokenizer)
        self.keys = list(tokenizer.vocab.keys())
        index_to_pop = self.keys.index('Ġ')
        self.keys.pop(index_to_pop)

        stripped = np.vectorize(lambda x: x.strip('Ġ '))
        # self.keys = np.array(stripped(self.keys))
        self.keys = np.array(self.keys)

        # create an array of ids
        # also remove the 'Ġ' token
        self.values = list(tokenizer.vocab.values())
        self.values.pop(index_to_pop)
        self.values = np.array(self.values)

        # dictionary mapping token value to key/value list index
        self.value_to_idx = {val: idx for idx, val in enumerate(self.values)}
        self.word_to_value = {word: self.values[idx] for idx, word in enumerate(self.keys)}

        cs_cmu = lambda x: count_syllables_cmu(x)
        cs_cmu = np.vectorize(cs_cmu)

        self.add_on_words = [x for x in self.keys if x[0] != 'Ġ']
        self.add_on_words_idxs = [i for i, x in enumerate(self.keys) if x[0] != 'Ġ']

        self.syllable_values = []
        self.syllable_idxs = []
        for i in range(1, 10):
            idxs = np.where(cs_cmu(self.keys) == i)
            self.syllable_idxs.append(idxs[0])
            self.syllable_values.append(self.values[idxs])

        # tokenizer.add_tokens('specialnewword')
        # self.extra_token = tokenizer('specialnewword')['input_ids'][0]
        self.extra_token = 19569  # Curt - poor Curt is sacrificed


    def __call__(self, input_ids, scores):
        banned_tokens = []
        # for every beam (partially generated sentence)
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            text = tokenizer.decode(beam_input_ids)
            word_syllables = get_text_syllables(text)
            last_rhyme_word_index = get_last_syllable_breakpoint(word_syllables, self.n_syllables)
            if last_rhyme_word_index:
                last_rhyme_word = word_syllables[last_rhyme_word_index][0]
                rhyming_words = set(pronouncing.rhymes(last_rhyme_word))
                rhyming_words_idxs = np.array([self.value_to_idx[self.word_to_value[word]] for word in rhyming_words if word in self.word_to_value])
            else:
                rhyming_words = set([])
                rhyming_words_idxs = np.array([])
            num_syllables = get_text_num_syllables(text)
            # print(num_syllables)
            num_syllables = num_syllables % self.n_syllables

            to_ban = [idxs for i, idxs in enumerate(self.syllable_values) if i + 1 + num_syllables > self.n_syllables]

            just_fit_ban = []
            just_fit_idxs = self.syllable_idxs[self.n_syllables-num_syllables-1]
            just_fit_rhyming_intersection= np.intersect1d(rhyming_words_idxs, just_fit_idxs, assume_unique=True)
            just_fit_rhyming_intersection = set(just_fit_rhyming_intersection)  # indices within just_fit_idxs
            just_fit_idxs_ban = np.array([idx for idx in just_fit_idxs if not idx in just_fit_rhyming_intersection])

            to_ban.append(just_fit_idxs_ban)

            # catch add on words like ion or ions which take the whole count over.
            add_on_word_idxs_to_ban = []
            for j, add_on_word in enumerate(self.add_on_words):
                tokens = [self.word_to_value[add_on_word]]
                i = -1
                while True:
                    token = beam_input_ids[i]
                    word = tokenizer.decode(token)
                    tokens.append(token)
                    if word[0] == ' ' and len(word) > 1:
                        break
                    i -= 1
                constructed = tokenizer.decode(tokens[::-1])
                constructed_previous = tokenizer.decode(tokens[1:][::-1])
                delta_syllables = get_text_num_syllables(constructed) - get_text_num_syllables(constructed_previous)
                bad = num_syllables - delta_syllables < 0 or num_syllables + delta_syllables > self.n_syllables
                if bad:
                    add_on_word_idxs_to_ban.append(self.add_on_words_idxs[i])
                if delta_syllables + num_syllables == self.n_syllables:
                    if not constructed in rhyming_words:
                        add_on_word_idxs_to_ban.append(self.add_on_words_idxs[i])

            to_ban.append(np.array(add_on_word_idxs_to_ban))

            if to_ban:
                print(f'num_syllables: {num_syllables}')
                print(text)
                print(quick_neatify_text(text))
                print(len(to_ban))
            banned_tokens.append(np.concatenate(to_ban) if to_ban else [])

        scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores


if __name__ == '__main__':

    # fsa = three_consecutive_word_FSA()
    # # fsa = even_FSA()
    # # instantiating a list of LogitsProcessor instances
    # # using our custom ABCLogits class
    # logits_processor = LogitsProcessorList([ConsecutiveLogits(fsa=fsa, vocab=tokenizer.vocab)])
    #
    # # running beam search using our custom LogitsProcessor
    # generated = model.beam_search(
    #     torch.cat([prompt_tokenized] * num_beams),
    #     beam_scorer,
    #     logits_processor=logits_processor,
    #     stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=12)])
    # )
    #
    # # printing the output beams
    # for index, output_tokenized in enumerate(generated):
    #     output = tokenizer.decode(output_tokenized)
    #     print(f'beam {index}: {output}')

    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        StoppingCriteriaList,
        MaxLengthCriteria
    )
    import torch

    # how many beams to track during the Viterbi algorithm
    num_beams = 5
    # how many beams to return after the algorithm
    num_return_beams = 5

    # the prompt to continue
    prompt = 'The wind rushed towards the man who lifted the spear from'

    # tokenizing the prompt
    prompt_tokenized = tokenizer(prompt, return_tensors='pt')
    prompt_tokenized = prompt_tokenized['input_ids']

    # instantiating a BeamSearchScorer
    beam_scorer = BeamSearchScorer(
        batch_size=prompt_tokenized.shape[0],
        num_beams=num_beams,
        num_beam_hyps_to_keep=num_return_beams,
        device=model.device
    )

    # instantiating a list of LogitsProcessor instances
    # using our custom ABCLogits class
    # logits_processor = LogitsProcessorList([ABCLogits(tokenizer.vocab)])
    # logits_processor = LogitsProcessorList([n_word_lines2(n_syllables=7)])
    logits_processor = LogitsProcessorList([long_word_encourager()])

    # running beam search using our custom LogitsProcessor
    # generated = model.beam_search(
    #     torch.cat([prompt_tokenized] * num_beams),
    #     beam_scorer,
    #     no_repeat_ngram_size=2,
    #     logits_processor = logits_processor,
    #     stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
    # )

    # # printing the output beams
    # for index, output_tokenized in enumerate(generated):
    #   output = tokenizer.decode(output_tokenized)
    #   print(f'beam {index}: {output}')

    outputs = model.generate(
        input_ids=prompt_tokenized,
        num_return_sequences=5,
        no_repeat_ngram_size=2,
        remove_invalid_values=True,
        logits_processor=logits_processor,
        max_length=50,
        do_sample=True,
        top_p=0.92,
        top_k=0
        # stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

