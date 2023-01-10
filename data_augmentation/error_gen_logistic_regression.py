from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from joblib import dump, load, Parallel, delayed
import data_augmentation.needleman_wunsch_alignment as align
import numpy as np
import torch

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class ErrorGenerator(object):

    match_idx = 0
    replace_idx = 1
    insert_idx = 2
    delete_index = 3

    def __init__(self, smoothing=4, simple_error_rate=0.05, parallel=1, simple=False, models_fpath=None, labeled_data=None):
        self.smoothing = smoothing
        self.simple_error_rate = simple_error_rate
        self.simple = simple
        self.parallel = parallel
        self.simple_probs = [1/3, 1/3, 1/3]

        if labeled_data is None:
            models = load(models_fpath)
            self.enc = models['one_hot_encoder']
            self.enc_labels = models['labels_encoder']
            self.regression = models['logistic_regression']
        elif models_fpath is None:
            X, Y = labeled_data
            self.enc = preprocessing.OneHotEncoder(sparse=True, handle_unknown="ignore")
            self.enc.fit(X)
            self.enc_labels = preprocessing.LabelEncoder()
            self.enc_labels.fit(Y)
            X_one_hot = self.enc.transform(X)
            Y_labels = self.enc_labels.transform(Y)
            self.regression = LogisticRegression(max_iter=100, solver='sag', tol=0.01).fit(X_one_hot, Y_labels)
 
        else:
            raise ValueError('cannot supply training data with path to model in constructor')

    def get_simple_synthetic_error_sequence(self, seq):
        err_prob = self.simple_error_rate
        synthetic_error_alignment = np.random.choice(
            [self.match_idx, self.replace_idx, self.insert_idx, self.delete_index],
            size=len(seq),
            replace=True,
            p=[
                1 - err_prob,
                err_prob * self.simple_probs[0],
                err_prob * self.simple_probs[1],
                err_prob * self.simple_probs[2],
                ]
            )

        # if by chance we end up with no errors, force an error for compliance
        if all(synthetic_error_alignment == self.match_idx):
            synthetic_error_alignment[len(synthetic_error_alignment) // 2] = self.replace_idx

        # get number of valid words from encoder
        vocab_size = len(self.enc.get_feature_names_out())

        errored_seq = []
        i = 0  # seq position
        for next_label in synthetic_error_alignment:
            if len(errored_seq) >= len(seq):
                break
            elif next_label == self.match_idx: # MATCH
                errored_seq.append(int(seq[i]))
                i += 1
            elif next_label == self.replace_idx: # REPLACE
                rand_mod = np.random.randint(vocab_size)
                errored_seq.append(rand_mod)                
                i += 1
            elif next_label == self.insert_idx: # INSERT
                rand_mod = np.random.randint(vocab_size)
                errored_seq.append(rand_mod)
            elif next_label == self.delete_index: # DELETE
                i += 1

        return errored_seq, list(synthetic_error_alignment)

    def get_synthetic_error_sequence(self, seq):
        errored_seq = []
        X_one_hot = self.enc.transform(np.array(seq).reshape(-1, 1))
        predictions = self.regression.predict_proba(X_one_hot)

        smooth_ind = int(np.median(np.argmax(predictions, 1)))
        # smooth predictions to reduce overall chance of errors
        predictions[:, smooth_ind] *= self.smoothing

        labels = []
        for p in predictions:
            p = p / np.sum(p)
            labels.append(np.random.choice(len(p), p=p))
        
        # recall that the output of the inverse transform here
        # is a string of 'operation.applicable vocab element'
        instructions = self.enc_labels.inverse_transform(labels)
        err_instructions = np.array([int(x[0]) for x in instructions])
        ind_instructions = np.array([int(x[2:]) for x in instructions])

        if sum(err_instructions != 3) < 3:
            err_instructions[0:3] = self.match_idx

        i = 0
        for j in range(len(err_instructions)):
            err_type = err_instructions[j]
            ind = ind_instructions[j]
            # err_type, ind = label
            if err_type == self.match_idx: # MATCH
                errored_seq.append(int(seq[i]))
                i += 1
            elif err_type == self.replace_idx: # REPLACE
                errored_seq.append(ind)                
                i += 1
            elif err_type == self.insert_idx: # INSERT
                errored_seq.append(int(seq[i]))
                errored_seq.append(ind)
                i += 1
            elif err_type == self.delete_index: # DELETE
                i += 1
        
        # fake_alignment = [x[0] for x in edit_instructions]
        return errored_seq, err_instructions

    def err_seq_string(self, labels):
        class_to_label = err_to_class = {0: 'O', 1: '~', 2: '+', 3: '-'}
        return ''.join(err_to_class[x] for x in labels)

    def add_errors_to_seq(self, inp, given_err_seq=None):
        inp = inp.astype('float32')

        seq_len = inp.shape[0]
        # X_out = np.zeros(inp.shape)
        # Y_out = np.zeros((inp.shape[0], inp.shape[1]))
        # inp = inp.numpy()

        Y_out = np.zeros(seq_len, dtype='float32')
        pad_seq = np.zeros(inp.shape, dtype='float32')

        # for n in range(X_out.shape[0]):
        orig_seq = list(inp)
        if given_err_seq is not None:
            err_seq = list(given_err_seq)
        elif self.simple:
            err_seq, err_record = self.get_simple_synthetic_error_sequence(orig_seq)
        else:
            err_seq, err_record = self.get_synthetic_error_sequence(orig_seq)

        assert len(err_seq) > 0, f"{err_seq}, {orig_seq} ,{err_record} ,{given_err_seq}"
        assert len(orig_seq) > 0, f"{err_seq} ,{orig_seq} ,{err_record} ,{given_err_seq}"

        _, _, r, _ = align.perform_alignment(orig_seq, err_seq, match_weights=[3, -2], gap_penalties=[-2, -2, -1, -1])

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

        return torch.from_numpy(X), torch.from_numpy(Y)

    def use_errors_from_existing_batch(self, batch):
        b, e = batch
        if not (type(b) == np.ndarray):
            b = batch.numpy()
            e = batch.numpy()

        out = [self.add_errors_to_seq(b[i], given_err_seq=e) for i in range(b.shape[0])]

        X = np.stack([np.array(x[0]) for x in out], 0)
        Y = np.stack([np.array(x[1]) for x in out], 0)

        return X, Y

    def save_models(self, fpath):
        d = {
            'one_hot_encoder': self.enc,
            'labels_encoder': self.enc_labels,
            'logistic_regression': self.regression,
        }
        dump(d, fpath)

if __name__ == "__main__":

    from agnostic_omr_dataloader import AgnosticOMRDataset
    from torch.utils.data import DataLoader
    from data_management.vocabulary import Vocabulary

    dset_path = r'./processed_datasets/quartets_felix_omr_agnostic_bymeasure.h5'
    v = Vocabulary(load_from_file='./data_management/vocab.txt')

    seq_len = 256
    proportion = 0.2
    dset = AgnosticOMRDataset(dset_path, seq_len, v)

    dload = DataLoader(dset, batch_size=50)
    batches = []
    for i, x in enumerate(dload):
        batch, metadata = x
        print(i, len(batch))
        batches.append(batch)
        if i > 2:
            break
    

    print('creating error generator')
    e = ErrorGenerator(smoothing=5, parallel=1, models_fpath='processed_datasets\quartet_omr_error_models_bymeasure.joblib')

    synth_error = e.get_synthetic_error_sequence(batch[0].numpy())
    simple_error = e.get_simple_synthetic_error_sequence(batch[0].numpy())
    print('adding errors to entire batch...')
    for i in range(2):
        print(i)
        e.simple = False
        X, Y = e.add_errors_to_batch(batch.numpy())
        e.simple = True
        X, Y = e.add_errors_to_batch(batch.numpy())
        print(X.shape, Y.shape)