import mne
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def process(filename):
    raw = mne.io.read_raw_gdf(filename, eog=['EOG-left', 'EOG-central', 'EOG-right'], preload=True)
    print()
    print(raw.info)
    print()
    
    rawData = raw.get_data()
    print(f'data shape - {rawData.shape}')

    events = mne.events_from_annotations(raw)
    nanidx = events[0][events[0][:, 2] == 5][:, 0]
    print(nanidx)
    nanidx = nanidx[1:]
    newnan = []
    for n in nanidx:
        for i in range(100):
            newnan.append(n+i)
    
    # plt.figure()
    # plt.plot(rawData[:, newnan])
    # plt.title('this should be a straight line')
    # plt.show()

    rawData[:, newnan] = np.nan
    print(rawData.shape)

    eogs = rawData[22:25, :]
    means = np.nanmean(eogs, axis=1, keepdims=True)
    stds = np.nanstd(eogs, axis=1, keepdims=True)
    print(f'means - {means.flatten()}')
    print(f'stds - {stds.flatten()}')
    std_eogs = standardize(eogs)
    filt_eogs = std_eogs.copy()
    filt_eogs[np.abs(filt_eogs) < 2] = np.nan

    nonout = np.sum(np.isnan(filt_eogs)) 
    total = 3*filt_eogs.shape[1]
    print(f'Total outlier (2sigma) values - {total-nonout}, Total values - {total}, Outlier percentage - {100*(total-nonout)/total:.2f}%')

    keepidx = ~np.isnan(filt_eogs)
    keepidx = np.argwhere(keepidx)
    print(keepidx.shape)

    tmplist = [[], [], []]

    keepidx = keepidx.flatten()

    i = 0
    j = 1
    while j < keepidx.shape[0]:
        tmplist[keepidx[i]].append(keepidx[j])
        i += 2
        j += 2

    keepidx = tmplist
    print(len(keepidx[0]), len(keepidx[1]), len(keepidx[2]), len(keepidx[0])+len(keepidx[1])+len(keepidx[2]))

    srate = raw.info['sfreq']
    seg_t = 2
    seg_n = int(seg_t*srate)
    half_seg = int(seg_n/2)
    curr_center = 0
    artefacts = []
    for i in range(3):
        curr_center = 0
        curr_edge = 0
        art = []
        for idx in keepidx[i]:
            if idx < curr_edge or idx-half_seg < curr_edge:
                continue
            else:
                curr_center = idx
                curr_edge = curr_center + half_seg
            
            if curr_center < half_seg:
                # take first two seconds
                tmp_artefact = eogs[i, :seg_n]
                curr_edge = seg_n + idx
                # print(f'{idx}\t0, {seg_n}')
            elif curr_center > eogs.shape[1] - half_seg:
                # take last two seconds
                tmp_artefact = eogs[i, -seg_n:]
                # print(f'{idx}\t{eogs.shape[1]-seg_n}, {eogs.shape[1]}')
                break
            else:
                # take two seconds around the center
                tmp_artefact = eogs[i, curr_center-half_seg:curr_center+half_seg]
                # print(f'{idx}\t{curr_center-half_seg}, {curr_center+half_seg}')
            art.append(tmp_artefact)
        artefacts.append(art)

    artefacts = artefacts[0] + artefacts[1] + artefacts[2]
    artefacts = np.array(artefacts)
    print(f'artefacts shape - {artefacts.shape}')

    # process eegs
    segs = []
    firsttwo = []
    annotations = raw.annotations
    onset_samples = annotations.onset
    dur_samples = annotations.duration
    for onset, dur, desc in zip(onset_samples, dur_samples, annotations.description):
        
        if desc == '768':
            startidx = raw.time_as_index(onset)[0]
            endidx = raw.time_as_index(onset + 2)[0]
            temp = rawData[:22, startidx:endidx]
            temp = temp[:, :500 * (temp.shape[1]//500)]
            temp = temp.reshape(-1, 500)

            # here, since we are taking exact 2s segments, we should set nans to 0 instead of deleting
            temp[np.isnan(temp)] = 0

            segs.append(temp)
        elif desc == '276' or desc == '277':
            startidx = raw.time_as_index(onset)[0]
            endidx = raw.time_as_index(onset + dur)[0]
            # print(startidx[0], endidx[0])
            # split into 2s segments
            # verify this later
            eegs = rawData[:22, startidx:endidx]
            # here we can delete the nans since we are creating the segments later.
            eegs = eegs[:, ~np.isnan(eegs).any(axis=0)]
            eegs = eegs[:, :500 * (eegs.shape[1]//500)]
            eegs = eegs.reshape(-1, 500)
            firsttwo.append(eegs)

    total768 = np.concatenate(segs, axis=0)
    totalidle = np.concatenate(firsttwo, axis=0)

    eegs = np.concatenate((total768, totalidle), axis=0)
    print(f'eegs shape - {eegs.shape}')

    return eegs, artefacts

def standardize(x):
    return (x - np.nanmean(x, axis=1, keepdims=True))/np.nanstd(x, axis=1, keepdims=True)

eegshapes = []
artshapes = []

eegs_array = []
artefacts_array = []

if __name__ == '__main__':
    for i in range(1, 10):
        for typ in ['T', 'E']:
            if i == 4 and typ == 'T':
                continue

            filename = f'datasets/BCI4/A0{i}{typ}.gdf'
            eegs, artefacts = process(filename)
            eegshapes.append(eegs.shape)
            artshapes.append(artefacts.shape)

            eegs_array.append(eegs)
            artefacts_array.append(artefacts)
            print('-----------------------------------------')
            print()
    
    print(eegshapes)
    print(artshapes)

    eegs_np = np.concatenate(eegs_array, axis=0)
    artefacts_np = np.concatenate(artefacts_array, axis=0)

    print(eegs_np.shape)
    print(artefacts_np.shape)

    np.save('datasets/BCI4/eegs_BCI.npy', eegs_np)
    np.save('datasets/BCI4/artefacts_BCI.npy', artefacts_np)