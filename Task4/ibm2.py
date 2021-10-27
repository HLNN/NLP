from collections import defaultdict
from tqdm import tqdm

from nltk.translate import Alignment
from nltk.translate.ibm_model import Counts


class IBMModel2:
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
            lambda: defaultdict(lambda: IBMModel2.MIN_PROB)
        )

        self.alignment_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: IBMModel2.MIN_PROB)))
        )

        if probability_tables is None:
            # Get translation probabilities from IBM Model 1
            # Run more iterations of training for Model 1, since it is
            # faster than Model 2
            # ibm1 = IBMModel1(sentence_aligned_corpus, 2 * iterations)
            # self.translation_table = ibm1.translation_table
            self.set_uniform_probabilities(sentence_aligned_corpus)
        else:
            # Set user-defined probabilities
            self.translation_table = probability_tables["translation_table"]
            self.alignment_table = probability_tables["alignment_table"]

        for n in tqdm(range(0, iterations)):
            self.train(sentence_aligned_corpus)

        self.align_all(sentence_aligned_corpus)

    def set_uniform_probabilities(self, sentence_aligned_corpus):
        initial_prob = 1 / len(self.trg_vocab)
        for t in self.trg_vocab:
            self.translation_table[t] = defaultdict(lambda: initial_prob)

        # a(i | j,l,m) = 1 / (l+1) for all i, j, l, m
        l_m_combinations = set()
        for aligned_sentence in sentence_aligned_corpus:
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)
            if (l, m) not in l_m_combinations:
                l_m_combinations.add((l, m))
                initial_prob = 1 / (l + 1)
                for i in range(0, l + 1):
                    for j in range(1, m + 1):
                        self.alignment_table[i][j][l][m] = initial_prob

    def train(self, parallel_corpus):
        counts = Model2Counts()
        for aligned_sentence in parallel_corpus:
            src_sentence = [None] + aligned_sentence.mots
            trg_sentence = ["UNUSED"] + aligned_sentence.words  # 1-indexed
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)

            # E step (a): Compute normalization factors to weigh counts
            total_count = self.prob_all_alignments(src_sentence, trg_sentence)

            # E step (b): Collect counts
            for j in range(1, m + 1):
                t = trg_sentence[j]
                for i in range(0, l + 1):
                    s = src_sentence[i]
                    count = self.prob_alignment_point(i, j, src_sentence, trg_sentence)
                    normalized_count = count / total_count[t]

                    counts.update_lexical_translation(normalized_count, s, t)
                    counts.update_alignment(normalized_count, i, j, l, m)

        # M step: Update probabilities with maximum likelihood estimates
        self.maximize_lexical_translation_probabilities(counts)
        self.maximize_alignment_probabilities(counts)

    def maximize_lexical_translation_probabilities(self, counts):
        for t, src_words in counts.t_given_s.items():
            for s in src_words:
                estimate = counts.t_given_s[t][s] / counts.any_t_given_s[s]
                self.translation_table[t][s] = max(estimate, IBMModel2.MIN_PROB)

    def maximize_alignment_probabilities(self, counts):
        MIN_PROB = IBMModel2.MIN_PROB
        for i, j_s in counts.alignment.items():
            for j, src_sentence_lengths in j_s.items():
                for l, trg_sentence_lengths in src_sentence_lengths.items():
                    for m in trg_sentence_lengths:
                        estimate = (
                            counts.alignment[i][j][l][m]
                            / counts.alignment_for_any_i[j][l][m]
                        )
                        self.alignment_table[i][j][l][m] = max(estimate, MIN_PROB)

    def prob_all_alignments(self, src_sentence, trg_sentence):
        alignment_prob_for_t = defaultdict(lambda: 0.0)
        for j in range(1, len(trg_sentence)):
            t = trg_sentence[j]
            for i in range(0, len(src_sentence)):
                alignment_prob_for_t[t] += self.prob_alignment_point(
                    i, j, src_sentence, trg_sentence
                )
        return alignment_prob_for_t

    def prob_alignment_point(self, i, j, src_sentence, trg_sentence):
        l = len(src_sentence) - 1
        m = len(trg_sentence) - 1
        s = src_sentence[i]
        t = trg_sentence[j]
        return self.translation_table[t][s] * self.alignment_table[i][j][l][m]

    def prob_t_a_given_s(self, alignment_info):
        prob = 1.0
        l = len(alignment_info.src_sentence) - 1
        m = len(alignment_info.trg_sentence) - 1

        for j, i in enumerate(alignment_info.alignment):
            if j == 0:
                continue  # skip the dummy zeroeth element
            trg_word = alignment_info.trg_sentence[j]
            src_word = alignment_info.src_sentence[i]
            prob *= (
                self.translation_table[trg_word][src_word]
                * self.alignment_table[i][j][l][m]
            )

        return max(prob, IBMModel2.MIN_PROB)

    def align_all(self, parallel_corpus):
        for sentence_pair in parallel_corpus:
            self.align(sentence_pair)

    def align(self, sentence_pair):
        best_alignment = []

        l = len(sentence_pair.mots)
        m = len(sentence_pair.words)

        for j, trg_word in enumerate(sentence_pair.words):
            # Initialize trg_word to align with the NULL token
            best_prob = (
                self.translation_table[trg_word][None]
                * self.alignment_table[0][j + 1][l][m]
            )
            best_prob = max(best_prob, IBMModel2.MIN_PROB)
            best_alignment_point = None
            for i, src_word in enumerate(sentence_pair.mots):
                align_prob = (
                    self.translation_table[trg_word][src_word]
                    * self.alignment_table[i + 1][j + 1][l][m]
                )
                if align_prob >= best_prob:
                    best_prob = align_prob
                    best_alignment_point = i

            best_alignment.append((j, best_alignment_point))

        sentence_pair.alignment = Alignment(best_alignment)


class Model2Counts(Counts):
    def __init__(self):
        super().__init__()
        self.alignment = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        )
        self.alignment_for_any_i = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0.0))
        )

    def update_lexical_translation(self, count, s, t):
        self.t_given_s[t][s] += count
        self.any_t_given_s[s] += count

    def update_alignment(self, count, i, j, l, m):
        self.alignment[i][j][l][m] += count
        self.alignment_for_any_i[j][l][m] += count


if __name__ == '__main__':
    from utils import *
    corpus = load_data()
    print("Train...")
    imb2 = IBMModel2(corpus, 16)
    pass
