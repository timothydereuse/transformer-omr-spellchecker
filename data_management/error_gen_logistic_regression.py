from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder(sparse=False)
enc.fit(X)
X_one_hot = enc.transform(X)
regression = LogisticRegression(max_iter=1000).fit(X_one_hot, Y)


def get_synthetic_error_sequence(seq, regression):
    ngram = 3
    num_ops = 4
    ins_idx = 3

    seq = np.array(seq)
    last_labels = [0 for _ in range(ngram)]

    # assemble note to run regression on: one note from the sequence and 3 previous labels
    
    i = 0
    while i < len(seq):
        next_note = np.concatenate([last_labels[-ngram:], seq[i]])
        predictions = regression.predict_proba(enc.transform(next_note.reshape(1, -1)))[0]
        next_label = np.random.choice(num_ops, p=predictions)
        last_labels.append(next_label)

        # if inserting into this sequence, then the previous note of the correct sequence stays the same.
        if not next_label == ins_idx:
            i += 1
    
    return last_labels


    # pad = np.zeros((seq.shape[0], 3)) # because trigrams
    # seq_arr = np.concatenate([pad, seq], 1)


