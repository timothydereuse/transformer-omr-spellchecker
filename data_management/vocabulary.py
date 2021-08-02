class Vocabulary(object):

    SEQ_UNK = 0
    SEQ_PAD = 1
    SEQ_SOS = 2
    SEQ_EOS = 3

    wtv = {}
    vtw = {}

    num_words = 0

    def __init__(self, c=None, load_from_file=None):
        if load_from_file:
            self.load_vocabulary(load_from_file)
        elif c:
            self.wtv['SEQ_UNK'] = self.SEQ_UNK
            self.vtw[self.SEQ_UNK] = 'SEQ_UNK'
            self.wtv['SEQ_PAD'] = self.SEQ_PAD
            self.vtw[self.SEQ_PAD] = 'SEQ_PAD'
            self.wtv['SEQ_SOS'] = self.SEQ_SOS
            self.vtw[self.SEQ_SOS] = 'SEQ_SOS'
            self.wtv['SEQ_EOS'] = self.SEQ_EOS
            self.vtw[self.SEQ_EOS] = 'SEQ_EOS'
            self.num_words = 4
            self.update(c)
    
    def update(self, c, min_freq=5):
        for w in sorted(c.keys(), key=lambda x: c[x], reverse=True):
            if c[w] < min_freq:
                continue
            self.wtv[w] = self.num_words
            self.vtw[self.num_words] = w
            self.num_words += 1

    def save_vocabulary(self, fname):
        lines = []
        for word in self.wtv.keys():
            lines.append(f'{word},{self.wtv[word]}\n')

        with open(fname, 'a') as f:
            for l in lines:
                f.write(l)

    def load_vocabulary(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()

        for l in lines:
            s = l.split(',')
            word = s[0].strip('\n')
            vec = int(s[1].strip('\n'))
            self.wtv[word] = vec
            self.vtw[vec] = word




if __name__ == '__main__':
    import data_management.semantic_to_agnostic as sta
    import music21 as m21
    import os
    from collections import Counter

    keys = ['felix', 'ABC', 'felix_errors', 'kernscores']
    quartets_root = r'D:\Documents\datasets\just_quartets'
    all_tokens = Counter()

    for k in keys:
        files = os.listdir(os.path.join(quartets_root, k))
        for fname in files:
            print(f'processing {fname}')
            fpath = os.path.join(os.path.join(quartets_root, k, fname))
            parsed_file = m21.converter.parse(fpath)
            parts = list(parsed_file.getElementsByClass(m21.stream.Part))
            # part = parts[0].getElementsByClass(m21.stream.Measure)
            print(f'processing {k}.{fname}')
            print(f'ntokens {len(all_tokens)}')

            for p in parts:
                agnostic = sta.m21_part_to_agnostic(p)
                all_tokens.update(agnostic)
    
    v = Vocabulary(all_tokens)
    v.save_vocabulary('./data_management/vocab.txt') 
    vv = Vocabulary(load_from_file='./data_management/vocab.txt')