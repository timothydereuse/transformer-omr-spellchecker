from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np

class ErrorGenerator(object):

    match_idx = 0
    replace_idx = 1
    insert_idx = 2
    delete_index = 3

    def __init__(self, X, Y, ngram, ins_samples, repl_samples):
        self.enc = preprocessing.OneHotEncoder(sparse=True)
        self.enc.fit(X)
        X_one_hot = self.enc.transform(X)
        self.regression = LogisticRegression(max_iter=5000).fit(X_one_hot, Y)
        self.ngram = ngram
        self.ins_samples = ins_samples
        self.repl_samples = repl_samples

    def get_synthetic_error_sequence(self, seq):
        num_ops = 4

        seq = np.array(seq)
        errored_seq = []
        gen_labels = [0 for _ in range(self.ngram)]

        ins_samples = self.ins_samples
        repl_samples = self.repl_samples

        # assemble note to run regression on: one note from the sequence and 3 previous labels
        
        i = 0
        while i < len(seq):
            next_note = np.concatenate([gen_labels[-self.ngram:], seq[i]])
            predictions = self.regression.predict_proba(self.enc.transform(next_note.reshape(1, -1)))[0]
            next_label = np.random.choice(len(predictions), p=predictions)
            gen_labels.append(next_label)

            if next_label == self.match_idx: # MATCH
                errored_seq.append(seq[i])
                i += 1
            elif next_label == self.replace_idx: # REPLACE
                rand_mod = repl_samples[np.random.randint(len(repl_samples))]
                errored_seq.append(seq[i] + rand_mod)
                i += 1
            elif next_label == self.insert_idx: # INSERT
                rand_ins = ins_samples[np.random.randint(len(ins_samples))]
                errored_seq.append(rand_ins)
            elif next_label == self.delete_index: # DELETE
                i += 1
        
        return errored_seq, gen_labels[self.ngram:]

    def err_seq_string(self, labels):
        class_to_label = err_to_class = {0: 'O', 1: '~', 2: '+', 3: '-'}
        return ''.join(err_to_class[x] for x in labels)


    # pad = np.zeros((seq.shape[0], 3)) # because trigrams
    # seq_arr = np.concatenate([pad, seq], 1)


