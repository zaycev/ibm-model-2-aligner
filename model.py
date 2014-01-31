# coding: utf-8

import em
import os
import lz4
import pickle
import logging
import collections
import numpy as np


MAX_LEN = 1000000

class Serializable(object):

    def save(self):
        file_path = os.path.join("tmp", "%s.pz4" % self.name)
        with open(file_path, "wb") as fl:
            fl.write(lz4.compressHC(pickle.dumps(self)))
        logging.info("Saved %r to %s" % (self, file_path))

    @staticmethod
    def load(name):
        file_path = os.path.join("tmp", "%s.pz4" % name)
        with open(file_path, "rb") as fl:
            model = pickle.loads(lz4.decompress(fl.read()))
            logging.info("Loaded %r from %s." % (model, file_path))
            return model

class Model1(Serializable):

    def __init__(self, name, e_lex, f_lex):
        self.name = name
        self.E = e_lex
        self.F = f_lex
        self.T = dict()

    def fit(self, pcorpus, iterations=3, smooth_n=0):
        """
        Implements simple divide and count IBM Model 1 training without FB-algorithm.
        """
        v_uniform = em.DT(1) / em.DT(len(self.E))
        T = collections.Counter()

        for itr in xrange(1, iterations + 1):

            logging.info("%r fitting, %d iteration" % (self, itr))

            count = collections.Counter()
            total = collections.Counter()

            for i, (E, F) in enumerate(pcorpus.itervectors()):

                F = [0] + F
                s_total = collections.Counter()
                if i % 1000 == 0:
                    logging.info("Handled %d pairs (%d)." % (i, len(count)))

                # Compute normalization.
                for e in E:
                    for f in F:
                        if itr == 1:
                            t_ef = v_uniform
                        else:
                            t_ef = T[(e, f)]
                        s_total[e] += t_ef

                # Collect counts.
                for e in E:
                    for f in F:
                        if itr == 1:
                            t_ef = v_uniform
                        else:
                            t_ef = T[(e, f)]
                        v = t_ef / s_total[e]
                        count[(e, f)] += v
                        total[f] += v

            if itr == 1:
                for e_f in count.iterkeys():
                    count[e_f] += smooth_n
                    total[e_f[1]] += smooth_n

            # Estimate probabilities.
            logging.info("Counts table size = %d" % len(count))
            logging.info("Recomputing T probabilities.")
            j = 0
            for e_f, count_e_f in count.iteritems():
                T[e_f] = count_e_f / total[e_f[1]]

        for e_f, e_f_p in T.iteritems():
            e, f = e_f
            if e in self.T:
                self.T[e][f] = float(e_f_p)
            else:
                self.T[e] = collections.Counter()
                self.T[e][f] = float(e_f_p)

        logging.info("Iter %d done. Log perplexity: %f" % (itr, np.log(self.perplexity(pcorpus))))


        logging.info("Done.")

    def smooth(self, n=3):
        pass

    def pair_perplexity(self, E, F):
        """
        Implements perplexity formula from Koehn book (4.15-4.21)
        """
        p = 0.0
        for e in E:
            for f in F:
                p += self.T[e][f]
        p /= float(len(F) ** len(E))
        return np.log2(p)

    def perplexity(self, pcorpus):
        P = 0.0
        n = 0.0
        for EF in pcorpus.itervectors():
            P += self.pair_perplexity(*EF)
            n += 1
        return 2 ** (-1.0 / n * P)

    def __str__(self):
        return "<Model1('%s')>" % self.name

    def __repr__(self):
        return self.__str__()


class Model2(Serializable):

    def __init__(self, name, model_1):
        self.name = name
        self.T = model_1.T
        self.E = model_1.E
        self.F = model_1.F
        self.A = {}

    def fit(self, pcorpus, iterations=3):
        """
        Implements simple divide and count IBM Model 2 training without FB-algorithm.
        """
        A = {}
        T = self.T

        for itr in xrange(1, iterations + 1):

            logging.info("%r fitting, %d iteration" % (self, itr))

            count = collections.Counter()
            total = collections.Counter()
            count_a = collections.Counter()
            total_a = collections.Counter()

            for i, (E, F) in enumerate(pcorpus.itervectors()):

                if i % 1000 == 0:
                    logging.info("Handled %d pairs." % i)

                s_total = collections.Counter()

                L_e = len(E)
                L_f = len(F)

                # Compute normalization.
                for J in xrange(1, L_e + 1):
                    e_J = E[J - 1]
                    for I in xrange(0, L_f + 1):
                        if I == 0:
                            f_I = 0
                        else:
                            f_I = F[I - 1]
                        if itr == 1:
                            a = em.DT(1) / em.DT(L_f + 1)
                        else:
                            a = A[(I, J, L_e, L_f)]

                        t = T[e_J][f_I]
                        s_total[e_J] += t * a

                # Collect counts.
                for J in xrange(1, L_e + 1):
                    e_J = E[J - 1]
                    for I in xrange(0, L_f + 1):
                        if I == 0:
                            f_I = 0
                        else:
                            f_I = F[I - 1]
                        I_J_Le_Lf = (I, J, L_e, L_f)
                        if itr == 1:
                            a = em.DT(1) / em.DT(L_f + 1)
                        else:
                            a = A[(I, J, L_e, L_f)]
                        c = T[e_J][f_I] * a / s_total[e_J]

                        count[(e_J, f_I)] += c
                        total[f_I] += c
                        count_a[I_J_Le_Lf] += c
                        total_a[(J, L_e, L_f)] += c

            # Estimate probabilities.
            logging.info("Estimating T probabilities.")
            for e, TF in T.iteritems():
                for f, p in TF.iteritems():
                    T[e][f] = count[(e, f)] / total[f]

            logging.info("Estimating A probabilities.")
            for I_J_Le_Lf, c_a in count_a.iteritems():
                _, J, L_e, L_f = I_J_Le_Lf
                t_a = total_a[(J, L_e, L_f)]
                A[I_J_Le_Lf] = float(c_a / t_a)

        self.A = {}
        for k, v in A.iteritems():
            self.A[k] = float(v)

        self.T = {}
        for k1, v1 in T.iteritems():
            self.T[k1] = collections.Counter()
            for k2, v2 in v1.iteritems():
                self.T[k1][k2] = float(v2)

        logging.info("Iter %d done. Log perplexity: %f" % (itr, np.log(self.perplexity(pcorpus))))

        logging.info("Done.")

    def pair_perplexity(self, E, F):
        """
        Implements perplexity formula from Koehn book (4.15-4.21)
        """
        p = 0.0
        for e in E:
            for f in F:
                p += self.T[e][f]
        p /= float(len(F) ** len(E))
        return np.log2(p)

    def perplexity(self, pcorpus):
        P = 0.0
        n = 0.0
        for EF in pcorpus.itervectors():
            P += self.pair_perplexity(*EF)
            n += 1
        return 2 ** (-1.0 / n * P)


    def allign_sentences(self, pcorpus, size=2):
        # print "Alignment"
        # Es = ["my", "third", "point", "has", "also", "been", "mentioned", "already", "."]
        # Fs = ["der", "dritte", "punkt", "wurde", "auch", "schon", "erwähnt".encode("utf-8"), "."]
        # E = self.E.doc2bow(Es)
        # F = self.F.doc2bow(Fs)
        # align = self.viterbi_alignment(E, F)
        # print align
        # for f, e in align:
        #     print "%d (%d) ––– %d (%s)" % (f, Fs[f], e, Es[e])
        # exit(0)
        for i, (E, F) in enumerate(pcorpus.itervectors()):
            if i == size:
                break
            yield self.viterbi_alignment(E, F)

    def viterbi_alignment(self, E, F):
        """
        Alignment from F to E translation (same as in Koehn book).
        """
        alignment = []
        L_e = len(E)
        L_f = len(F)

        j = 0
        for e in E:
            j += 1

            best_i = 0
            best_p = -np.inf

            for (i, f) in enumerate([0] + F):
                a = np.log2(self.A.get((i, j, L_e, L_f), 0.0))
                t = np.log2(self.T.get(e, {}).get(f, 0.0))
                p = a + t
                if p > best_p:
                    best_i = i
                    best_p = p
            if best_i > 0:
                alignment.append((best_i - 1, j - 1))

        return alignment

    def __str__(self):
        return "<Model2('%s')>" % self.name

    def __repr__(self):
        return self.__str__()


class Alignment(object):

    def __init__(self, alignments):
        self.alignments = alignments

    @staticmethod
    def load(file_path, size=2):
        alignments = []
        possibly, sure = [], []
        with open(file_path, "rb") as fl:
            while True:
                try:
                    header = fl.readline().rstrip("\n")
                    a_id = int(header.split(" ")[1])
                    if a_id >= size:
                        break
                    while True:
                        line = fl.readline().rstrip("\n")
                        if len(line) == 0:
                            break
                        tag, f, e = line.split(" ")
                        if tag == "P":
                            possibly.append((int(f), int(e)))
                        elif tag == "S":
                            sure.append((int(f), int(e)))
                        else:
                            logging.error("Alignment reading error.")

                    alignments.append((set(possibly), set(sure)))
                    possibly, sure = [], []
                except:
                    import traceback
                    logging.error(traceback.format_exc())
                    return Alignment(alignments)

        return Alignment(alignments)

    def aer(self, pred_alignments):
        pred_alignments = list(pred_alignments)
        if len(self.alignments) != len(pred_alignments):
            logging.error("Alignments are diffirent size.")
        for i, A in enumerate(pred_alignments):
            A = set(A)
            P, S = self.alignments[i]
            yield 1 - float(len(A & S) + len(A & (P | S))) / float(len(A) + len(S))


class PCorpus(object):

    def __init__(self, e_path, f_path, e=None, f=None):
        self.e_path = e_path
        self.f_path = f_path
        self.e = e
        self.f = f

    def __iter__(self):
        with open(self.e_path, "rb") as e_fl, open(self.f_path, "rb") as f_fl:
            for i, e_line in enumerate(e_fl):
                if i > MAX_LEN:
                    break
                f_line = f_fl.readline()
                e_tokens = e_line.rstrip("\n").split(" ")
                f_tokens = f_line.rstrip("\n").split(" ")
                yield e_tokens, f_tokens

    def itervectors(self):
        for E, F in self:
            yield self.e.doc2bow(E), self.f.doc2bow(F)

    def __str__(self):
        return "<PCorpus(e=%s, f=%s)>" % (self.e_path, self.f_path)

    def __repr__(self):
        return self.__str__()


class Lexicon(Serializable):
    NULL_WORD = "<NULL>"


    def __init__(self, name):
        self.name = name
        self.w2f = collections.Counter()
        self.words = set()
        self.tf = 0
        self.df = 0
        self.w2id = {}
        self.id2w = {}

    def add_document(self, words):
        self.w2f.update(words)
        self.words.update(words)
        self.tf += len(words)
        self.df += 1

    def collect(self, file_path, add_null=False):
        with open(file_path, "rb") as fl:
            for i, line in enumerate(fl):
                if i > MAX_LEN:
                    break
                line = line.rstrip("\n")
                words = line.split(" ")
                self.add_document(words)
        if add_null:
            self.w2f[self.NULL_WORD] = 0
            self.words.add(self.NULL_WORD)
        return self

    def vectorize(self, add_null=False):
        for w in self.words:
            w_id = len(self.w2id) + 1
            self.w2id[w] = w_id
            self.id2w[w_id] = w
        if add_null:
            self.w2id[Lexicon.NULL_WORD] = 0
            self.id2w[0] = Lexicon.NULL_WORD

    def doc2bow(self, words):
        return [self.w2id[w] for w in words]

    def bow2doc(self, bow):
        return [self.id2w[i] for i in bow]

    def __iter__(self):
        return self.words.__iter__()

    def __getitem__(self, word):
        return self.w2f.get(word, 0.0)

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return word in self.words

    def __str__(self):
        return "<Lexicon('%s', words=%d, df=%d, tf=%d)>" % (
            self.name,
            len(self.words),
            self.df,
            self.tf)

    def __repr__(self):
        return self.__str__()