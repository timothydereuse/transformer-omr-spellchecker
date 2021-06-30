from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from joblib import dump, load
import data_management.needleman_wunsch_alignment as align
import numpy as np

class ErrorGenerator(object):

    match_idx = 0
    replace_idx = 1
    insert_idx = 2
    delete_index = 3

    def __init__(self, ngram, models_fpath=None, labeled_data=None, ins_samples=None, repl_samples=None):
        self.ngram = ngram
        if labeled_data is None and ins_samples is None and repl_samples is None:
            models = load(models_fpath)
            self.enc = models['one_hot_encoder']
            self.regression = models['logistic_regression']
            self.ins_samples = models['insert_samples']
            self.repl_samples = models['replace_samples']
        elif models_fpath is None:
            X, Y = labeled_data
            self.enc = preprocessing.OneHotEncoder(sparse=True)
            self.enc.fit(X)
            X_one_hot = self.enc.transform(X)
            self.regression = LogisticRegression(max_iter=5000).fit(X_one_hot, Y)
            self.ins_samples = ins_samples
            self.repl_samples = repl_samples
        else:
            raise ValueError('cannot supply training data with path to model in constructor')

    def get_synthetic_error_sequence(self, seq):

        num_ops = 4
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

    def save_models(self, fpath):
        d = {
            'one_hot_encoder': self.enc,
            'logistic_regression': self.regression,
            'insert_samples': self.ins_samples,
            'replace_samples': self.repl_samples
        }
        dump(d, fpath)

    def err_seq_string(self, labels):
        class_to_label = err_to_class = {0: 'O', 1: '~', 2: '+', 3: '-'}
        return ''.join(err_to_class[x] for x in labels)

    def add_errors_to_batch_parallel(self, batch, n_jobs=3, verbose=0):
        x = batch.numpy()
        out = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.add_errors_to_seq)(x[i]) for i in range(x.shape[0])
            )

        X = np.stack([np.array(x[0]) for x in out], 0)
        Y = np.stack([np.array(x[1]) for x in out], 1)

        return X, Y


    def add_errors_to_seq(self, inp):

        seq_len = inp.shape[0]
        X_out = np.zeros(inp.shape)
        # Y_out = np.zeros((inp.shape[0], inp.shape[1]))
        # inp = inp.numpy()

        Y_out = np.zeros(seq_len)
        pad_seq = np.zeros(inp.shape)

        # for n in range(X_out.shape[0]):
        orig_seq = list(inp)
        err_seq, _ = self.get_synthetic_error_sequence(orig_seq)

        _, _, r, _ = align.perform_alignment(orig_seq, err_seq, match_weights=[1, -1], gap_penalties=[-3, -3, -3, -3])

        # put 'deletion' markers in front of entries in alignment record r that record deletions
        res = np.zeros(seq_len)
        i = 0
        while i < len(r) and i < Y_out.shape[0]:
            if r[i] == '-':
                # r[i - 1] = 'D'
                Y_out[i - 1] = 1
                del r[i]
            elif r[i] == '~' or r[i] == '+':
                Y_out[i] = 1
                i += 1
            else:
                i += 1

        padded_seq = np.concatenate([err_seq, pad_seq], 0)[:seq_len, :]

        return padded_seq, Y_out

if __name__ == "__main__":

    from point_set_dataloader import MidiNoteTupleDataset
    from torch.utils.data import DataLoader


    fname = 'all_string_quartets.h5'
    seq_len = 500
    proportion = 0.2
    dset = MidiNoteTupleDataset(fname, seq_len, num_feats=4)

    dload = DataLoader(dset, batch_size=15)
    batches = []
    for i, x in enumerate(dload):
        print(i, x.shape)
        batches.append(x)
        if i > 2:
            break

    e = ErrorGenerator(ngram=5, models_fpath='savemodels.joblib')

    asdf = e.add_errors_to_seq(x[0].numpy())

    from joblib import Parallel, delayed

    # print('adding errors to entire batch...')
    X, Y = e.add_errors_to_batch_parallel(x)