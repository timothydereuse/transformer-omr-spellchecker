from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from joblib import dump, load, Parallel, delayed
import data_augmentation.needleman_wunsch_alignment as align
import numpy as np

class ErrorGenerator(object):

    match_idx = 0
    replace_idx = 1
    insert_idx = 2
    delete_index = 3

    def __init__(self, ngram, smoothing=1, simple_error_rate=0.05, parallel=1, simple=False, models_fpath=None, labeled_data=None, ins_samples=None, repl_samples=None):
        self.ngram = ngram
        self.smoothing = smoothing
        self.simple_error_rate = simple_error_rate
        self.simple = simple
        self.parallel = parallel

        if labeled_data is None and ins_samples is None and repl_samples is None:
            models = load(models_fpath)
            self.enc = models['one_hot_encoder']
            self.regression = models['logistic_regression']
            self.ins_samples = models['insert_samples']
            self.repl_samples = models['replace_samples']
            self.ngram = models['ngram']
        elif models_fpath is None:
            X, Y = labeled_data
            self.enc = preprocessing.OneHotEncoder(sparse=True, handle_unknown="ignore")
            self.enc.fit(X)
            X_one_hot = self.enc.transform(X)
            self.regression = LogisticRegression(max_iter=5000).fit(X_one_hot, Y)
            self.ins_samples = ins_samples
            self.repl_samples = repl_samples
        else:
            raise ValueError('cannot supply training data with path to model in constructor')

    def get_simple_synthetic_error_sequence(self, seq):
        err_prob = self.simple_error_rate
        synthetic_error_alignment = np.random.choice(
            [self.match_idx, self.replace_idx, self.insert_idx, self.delete_index],
            size=len(seq),
            replace=True,
            p=[1 - err_prob, err_prob / 3, err_prob / 3, err_prob / 3]
            )

        ins_samples = self.ins_samples
        repl_samples = self.repl_samples
        errored_seq = []
        i = 0  # seq position
        for next_label in synthetic_error_alignment:
            if len(errored_seq) >= len(seq):
                break
            elif next_label == self.match_idx: # MATCH
                errored_seq.append(seq[i].astype('float32'))
                i += 1
            elif next_label == self.replace_idx: # REPLACE
                rand_mod = repl_samples[np.random.randint(len(repl_samples))]
                errored_seq.append((rand_mod).astype('float32'))                
                i += 1
            elif next_label == self.insert_idx: # INSERT
                rand_ins = ins_samples[np.random.randint(len(ins_samples))]
                errored_seq.append(rand_ins.astype('float32'))
            elif next_label == self.delete_index: # DELETE
                i += 1

        return errored_seq, synthetic_error_alignment


    def get_synthetic_error_sequence(self, seq):
        num_ops = 4
        errored_seq = []
        gen_labels = [0 for _ in range(self.ngram)]

        ins_samples = self.ins_samples
        repl_samples = self.repl_samples

        # assemble note to run regression on: one note from the sequence and [ngram] previous labels
        
        i = 0
        while i < len(seq):
            next_note = np.concatenate([gen_labels[-self.ngram:], [seq[i]] ])
            predictions = self.regression.predict_proba(self.enc.transform(next_note.reshape(1, -1)))[0]

            # smooth predictions to reduce overall chance of errors
            smooth_dist = (1 - predictions[0]) * (1 - self.smoothing)
            predictions[0] += smooth_dist

            error_target = 1 - predictions[0]
            pred_sum = sum(predictions[1:])
            for j in range(1, len(predictions)):
                predictions[j] = predictions[j] / pred_sum * error_target

            next_label = np.random.choice(len(predictions), p=predictions)
            gen_labels.append(next_label)

            if next_label == self.match_idx: # MATCH
                errored_seq.append(seq[i].astype('float32'))
                i += 1
            elif next_label == self.replace_idx: # REPLACE
                rand_mod = repl_samples[np.random.randint(len(repl_samples))]
                errored_seq.append((rand_mod).astype('float32'))                
                i += 1
            elif next_label == self.insert_idx: # INSERT
                rand_ins = ins_samples[np.random.randint(len(ins_samples))]
                errored_seq.append(rand_ins.astype('float32'))
            elif next_label == self.delete_index: # DELETE
                i += 1
        
        return errored_seq, gen_labels[self.ngram:]

    def err_seq_string(self, labels):
        class_to_label = err_to_class = {0: 'O', 1: '~', 2: '+', 3: '-'}
        return ''.join(err_to_class[x] for x in labels)

    def add_errors_to_seq(self, inp):
        inp = inp.astype('float32')

        seq_len = inp.shape[0]
        # X_out = np.zeros(inp.shape)
        # Y_out = np.zeros((inp.shape[0], inp.shape[1]))
        # inp = inp.numpy()

        Y_out = np.zeros(seq_len, dtype='float32')
        pad_seq = np.zeros(inp.shape, dtype='float32')

        # for n in range(X_out.shape[0]):
        orig_seq = list(inp)
        if self.simple:
            err_seq, _ = self.get_simple_synthetic_error_sequence(orig_seq)
        else:
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

        padded_seq = np.concatenate([err_seq, pad_seq], 0)[:seq_len]

        return padded_seq, Y_out

    def add_errors_to_batch(self, batch, verbose=0):
        if not (type(batch) == np.ndarray):
            batch = batch.numpy()
        b = batch.astype('float32')
        
        if self.simple:
            out = [self.add_errors_to_seq(b[i]) for i in range(b.shape[0])]
        elif self.parallel >= 2:
            out = Parallel(n_jobs=self.parallel, verbose=verbose)(
                delayed(self.add_errors_to_seq)(b[i]) for i in range(b.shape[0])
                )
        else:
            out = [self.add_errors_to_seq(b[i]) for i in range(b.shape[0])]

        X = np.stack([np.array(x[0]) for x in out], 0)
        Y = np.stack([np.array(x[1]) for x in out], 0)

        return X, Y

    def save_models(self, fpath):
        d = {
            'one_hot_encoder': self.enc,
            'logistic_regression': self.regression,
            'insert_samples': self.ins_samples,
            'replace_samples': self.repl_samples,
            'ngram': self.ngram
        }
        dump(d, fpath)

if __name__ == "__main__":

    from agnostic_omr_dataloader import AgnosticOMRDataset
    from torch.utils.data import DataLoader
    from data_management.vocabulary import Vocabulary

    dset_path = r'./processed_datasets/quartets_felix_omr_agnostic.h5'
    v = Vocabulary(load_from_file='./data_management/vocab.txt')

    seq_len = 50
    proportion = 0.2
    dset = AgnosticOMRDataset(dset_path, seq_len, v)

    dload = DataLoader(dset, batch_size=5)
    batches = []
    for i, x in enumerate(dload):
        print(i, x.shape)
        batches.append(x)
        if i > 2:
            break

    print('creating error generator')
    e = ErrorGenerator(ngram=5, smoothing=0.7, parallel=3, models_fpath='./data_augmentation/quartet_omr_error_models.joblib')

    synth_error = e.get_synthetic_error_sequence(x[0].numpy())
    simple_error = e.get_simple_synthetic_error_sequence(x[0].numpy())
    print('adding errors to entire batch...')
    for i in range(2):
        print(i)
        X, Y = e.add_errors_to_batch(x.numpy())
        print(X.shape, Y.shape)