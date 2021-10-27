from collections import defaultdict
from tqdm import tqdm

from nltk.translate import Alignment
from nltk.translate.ibm_model import Counts


class IBMModel1:
    MIN_PROB = 1.0e-12

    def __init__(self, sentence_aligned_corpus, iterations, probability_tables=None):
        src_vocab = set()
        trg_vocab = set()
        for aligned_sentence in sentence_aligned_corpus:
            trg_vocab.update(aligned_sentence.words)
            src_vocab.update(aligned_sentence.mots)
        # Add the NULL token
        src_vocab.add(None)

        # Set of al source and target language word used in training
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.translation_table = defaultdict(
            lambda: defaultdict(lambda: IBMModel1.MIN_PROB)
        )

        self.alignment_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: IBMModel1.MIN_PROB)))
        )

        if probability_tables is None:
            self.set_uniform_probabilities(sentence_aligned_corpus)
        else:
            # Set user-defined probabilities
            self.translation_table = probability_tables["translation_table"]

        for n in tqdm(range(0, iterations)):
            self.train(sentence_aligned_corpus)

        self.align_all(sentence_aligned_corpus)

    def set_uniform_probabilities(self, sentence_aligned_corpus):
        initial_prob = 1 / len(self.trg_vocab)
        for t in self.trg_vocab:
            self.translation_table[t] = defaultdict(lambda: initial_prob)

    def train(self, parallel_corpus):
        counts = Counts()
        for aligned_sentence in parallel_corpus:
            trg_sentence = aligned_sentence.words
            src_sentence = [None] + aligned_sentence.mots

            # E step (a): Compute normalization factors to weigh counts
            total_count = self.prob_all_alignments(src_sentence, trg_sentence)

            # E step (b): Collect counts
            for t in trg_sentence:
                for s in src_sentence:
                    count = self.prob_alignment_point(s, t)
                    normalized_count = count / total_count[t]
                    counts.t_given_s[t][s] += normalized_count
                    counts.any_t_given_s[s] += normalized_count

        # M step: Update probabilities with maximum likelihood estimate
        self.maximize_lexical_translation_probabilities(counts)

    def maximize_lexical_translation_probabilities(self, counts):
        for t, src_words in counts.t_given_s.items():
            for s in src_words:
                estimate = counts.t_given_s[t][s] / counts.any_t_given_s[s]
                self.translation_table[t][s] = max(estimate, IBMModel1.MIN_PROB)

    def prob_all_alignments(self, src_sentence, trg_sentence):
        alignment_prob_for_t = defaultdict(lambda: 0.0)
        for t in trg_sentence:
            for s in src_sentence:
                alignment_prob_for_t[t] += self.prob_alignment_point(s, t)
        return alignment_prob_for_t

    def prob_alignment_point(self, s, t):
        return self.translation_table[t][s]

    def prob_t_a_given_s(self, alignment_info):
        prob = 1.0
        for j, i in enumerate(alignment_info.alignment):
            if j == 0:
                continue  # skip the dummy zeroeth element
            trg_word = alignment_info.trg_sentence[j]
            src_word = alignment_info.src_sentence[i]
            prob *= self.translation_table[trg_word][src_word]
        return max(prob, IBMModel1.MIN_PROB)

    def align_all(self, parallel_corpus):
        for sentence_pair in parallel_corpus:
            self.align(sentence_pair)

    def align(self, sentence_pair):
        best_alignment = []

        for j, trg_word in enumerate(sentence_pair.words):
            # Initialize trg_word to align with the NULL token
            best_prob = max(self.translation_table[trg_word][None], IBMModel1.MIN_PROB)
            best_alignment_point = None
            for i, src_word in enumerate(sentence_pair.mots):
                align_prob = self.translation_table[trg_word][src_word]
                if align_prob >= best_prob:  # prefer newer word in case of tie
                    best_prob = align_prob
                    best_alignment_point = i

            best_alignment.append((j, best_alignment_point))

        sentence_pair.alignment = Alignment(best_alignment)


if __name__ == '__main__':
    from utils import *
    corpus = load_data()
    print("Train...")
    imb1 = IBMModel1(corpus, 16)
    pass
