# coding: utf-8

import os
import logging
import numpy as np

from model import Model1
from model import Model2
from model import Lexicon
from model import PCorpus
from model import Alignment


logging.basicConfig(level=logging.INFO)



ROOT_DIR = "/Users/zvm/Dropbox/usc/2014-S/csci599/hw1"
DATA_ENG = os.path.join(ROOT_DIR, "train.clean.en")
DATA_GER = os.path.join(ROOT_DIR, "train.clean.de")
TEST_AGM = os.path.join(ROOT_DIR, "alignmentDeEn")

lex_e = Lexicon("eng.test").collect(DATA_ENG, add_null=True)
lex_g = Lexicon("ger.test").collect(DATA_GER, add_null=True)
lex_e.vectorize(add_null=True)
lex_g.vectorize(add_null=True)
lex_e.save()
lex_g.save()

# lex_e = Lexicon.load("eng")
# lex_g = Lexicon.load("ger")
pcorpus = PCorpus(DATA_ENG, DATA_GER, lex_e, lex_g)


model_1 = Model1("ibm_1.test", lex_e, lex_g)
model_1.fit(pcorpus, iterations=7, smooth_n=0.8)
model_1.save()
# model_1 = Model1.load("ibm_1.test")

model_2 = Model2("ibm_2.test", model_1)
model_2.fit(pcorpus, iterations=7)
model_2.save()
# model_2 = Model2.load("ibm_2.test")

# words = ["auto", "feuer", u"brücke".encode("utf-8"), "haus", "tabelle"]
# T = {}
# import collections
# for k1, v1 in model_2.T.iteritems():
#     for k2, v2 in v1.iteritems():
#         if k2 in T:
#             T[k2][k1] = v2
#         else:
#             T[k2] = collections.Counter()
#             T[k2][k1] = v2
# for w in words:
#     tw = T[model_2.F.w2id[w]].most_common(10)
#     print w
#     for w_id, w_p in tw:
#         print "\t", model_2.E.id2w[w_id], w_p
#     print

pred_alignments = model_2.allign_sentences(pcorpus, size=508)
test_alignments = Alignment.load(TEST_AGM, size=508)
aers = list(test_alignments.aer(pred_alignments))

print np.mean(aers)
