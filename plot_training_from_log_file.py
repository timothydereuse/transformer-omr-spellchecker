import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser(description='Prints nice graphs of training and validation progress from al')
parser.add_argument('logfile', default='', help='Log file to print training data from.')
args = vars(parser.parse_args())

log_file = args['logfile']

with open(log_file) as f:
    logs = f.readlines()

logs = [x.strip() for x in logs if ('train_loss' in x) and ('s/epoch' in x)]

# make empty dictionary to hold all entries
keys = logs[0].split('|')[1:]
keys = [re.split(r'\s+', k)[1] for k in keys if k]
dats = {k: [] for k in keys if k}

for line in logs:
    entries = line.split('|')[1:]
    for e in entries:
        spl = re.split(r'\s+', e)
        spl = [x for x in spl if x]
        if not spl:
            continue
        dats[spl[0]].append(float(spl[1]))

fig, axs = plt.subplots(2, 1, figsize=(12, 6))
epochs = np.arange(len(dats[keys[0]]))

axs[0].set_title('Training and Validation Loss')
axs[0].plot(epochs, dats['train_loss'], c='black', label='Training')
axs[0].plot(epochs, dats['val_loss'], c='red', label='Validation')
axs[0].set_xlim([-0.02, max(epochs)])
axs[0].set_ylim([-0.02, max(dats['val_loss'])])
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epoch')
axs[0].legend()

axs[1].set_title('F1 Score')
axs[1].plot(epochs, dats['tr_f1'], c='black', label='Training')
axs[1].plot(epochs, dats['val_f1'], c='red', label='Validation')
axs[1].set_xlim([0, max(epochs)])
axs[1].set_ylim([0, 1.02])
axs[1].set_ylabel('F1 Score')
axs[1].set_xlabel('Epoch')
axs[1].legend()

out_fname = f'{log_file[:-4]}_plot.png'
fig.savefig(out_fname, bbox_inches='tight')
plt.clf()
plt.close(fig)
