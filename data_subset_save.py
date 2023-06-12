import numpy as np 
import glob
import pickle


path = '/jukebox/witten/yousuf/rotation/pickles2/loop_files/'
files = glob.glob(path + '*')

fnum = 5
with open(files[fnum], 'rb') as handle:
    data = pickle.load(handle)
keys = data.keys()
# selecting the key that includes single and multi-unit recordings:
key = [key for i, key in enumerate(keys) if key.split("_")[1] == "all"]
data = data[key[0]]


# pick a mous  date:

mouseID = np.array(data["mouseID"])
print(f"mice: {np.unique(mouseID)}")
mouse_pick = 3  # pick mouse index here
mouse = np.unique(mouseID)[mouse_pick]

all_dates = np.asarray(data["date"])
# dates associated with that mouse:
dates = np.unique(all_dates[mouseID == mouse])
print(f"dates: {dates}")
date = dates[0]  # pick date index here

# index of mouse/dates of interest in data:
idx = np.nonzero((all_dates == date) * (mouseID == mouse))[0]
print(f"number of neurons for mouse {mouse} on {date}: {idx.size}")

out = dict()

out['nCues_RminusL'] = data['nCues_RminusL'][idx[0]]
out['currMaze'] = data['currMaze'][idx[0]]
out['laserON'] = data['laserON'][idx[0]]
out['trialStart'] = data['trialStart'][idx[0]]
out['trialEnd'] = data['trialEnd'][idx[0]]
out['keyFrames'] = data['keyFrames'][idx[0]]
out['time'] = data['time'][idx[0]]
out['cueOnset_L'] = data['cueOnset_L'][idx[0]]
out['cueOnset_R'] = data['cueOnset_R'][idx[0]]
out['choice'] = data['choice'][idx[0]]
out['trialType'] = data['trialType'][idx[0]]
out['pos'] = data['pos'][idx[0]]

out['spikes'] = [data['spikes'][i] for i in idx]
out['timeSqueezedFR'] = [data['timeSqueezedFR'][i] for i in idx]

with open('test_data_acc_ind_492_0607.pickle', 'wb') as handle:
    pickle.dump(out, handle)