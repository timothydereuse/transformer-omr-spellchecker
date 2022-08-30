import torch
import torch.nn as nn
import plot_outputs as po
import numpy as np
import sklearn.metrics
from torch.utils.data import DataLoader
import models.LSTUT_model as lstut
import model_params as params
from importlib import reload
import data_management.vocabulary as vocab
import os
import music21 as m21
import data_management.semantic_to_agnostic as sta

def make_pieces_dict(quartets_root, vocabulary):
    folders = []
    for k in ['felix', 'felix_errors_onepass']:
        files = os.listdir(os.path.join(quartets_root, k))
        files = sorted([f for f in files if 'op80' not in f])
        files = [os.path.join(quartets_root, k, fname) for fname in files]
        folders.append(files)
    pairs = list(zip(folders[0], folders[1])) 

    pieces = {}
    for pair in pairs:
        piece_record = {}

        for fpath, type in zip(pair, ['orig', 'error']):

            print(f'parsing {fpath}... ')
            file_record = {}
            parsed_file = m21.converter.parse(fpath)
            file_record['m21_stream'] = parsed_file
            parts = list(parsed_file.getElementsByClass(m21.stream.Part))
            file_record['m21_parts'] = parts
            agnostic = sta.m21_parts_to_interleaved_agnostic(parts, remove=['+'])
            file_record['agnostic'] = agnostic
            file_record['agnostic_encoded'] = vocabulary.words_to_vec(agnostic)
            piece_record[type] = file_record

        pieces[os.path.split(pair[0])[-1]] = piece_record

    return pieces

if __name__ == '__main__':

    quartets_root = r"D:\Documents\datasets\just_quartets"

    model_path = r'trained_models\lstut_best_LSTUT_FELIX_TRIAL_0_(2022.03.14.20.21)_1-1-1-01-0-32-32.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    params = checkpoint['params']

    v = vocab.Vocabulary(load_from_file=params.saved_vocabulary)

    pieces = make_pieces_dict(quartets_root, v)            

    lstut_settings = params.lstut_settings
    lstut_settings['vocab_size'] = v.num_words
    lstut_settings['seq_length'] = params.seq_length
    model = lstut.LSTUT(**lstut_settings).to(device)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.load_state_dict(checkpoint['model_state_dict'])
    # best_thresh = checkpoint['best_thresh']

    model.eval()