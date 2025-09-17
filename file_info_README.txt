data/ -> the standard EEGDenoiseNet data that is used as training input (30k train pairs, 4k test pairs).
data_bci/ -> our novel EOG+EEG dataset. _raw files are the original 2s 500 Hz EEG and EOG segments obtained from BCI-IV 2a, z-score standardised. _mini files are a small subset of the _raw files mixed at SNRs of -3 to +3 dB, upsampled to 512 Hz and ready-to-go. This is the data used for generalisation testing. Beware that _raw may have some nan values.
bci_processing_script.py -> this is used on raw BCI-IV data files (publicly available) to extract 2s EEG and EOG segments to construct the _raw files.
compile_metrics.ipynb -> performance metrics and their trends vs params can be visualised here.
datautils.py -> utility functions for handling data
main.py -> RUN THIS to train models
models.py -> holder for all model architectures
testviz.ipynb -> used to get quick visualisations of model performance - EEG v EOG+EEG v denoised signal plots.
utils.py -> general utility functions.

