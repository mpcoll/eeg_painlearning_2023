'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''

from mne.report import Report
import pprint
import mne
import os
from mne.preprocessing import ICA, create_eog_epochs
from os.path import join as opj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bids import BIDSLayout
from scipy.stats import zscore


###############################
# Parameters
##############################

inpath = '/media/mp/mpx6/2023_painlerarnign_validate/source'
outpath = '/media/mp/mpx6/2023_painlerarnign_validate/derivatives'

layout = BIDSLayout(inpath)
part = ['sub-' + s for s in layout.get_subject()]
param = {
    # EOG channels
    'eogchan': ['EXG3', 'EXG4', 'EXG5'],
    # Empty channels to drop
    'dropchan': ['EXG6', 'EXG7', 'EXG8'],
    # Channels to rename
    'renamechan': {'EXG1': 'M1', 'EXG2': 'M2', 'EXG3': 'HEOGL',
                   'EXG4': 'HEOGR', 'EXG5': 'VEOGL'},
    # # Participants to exclude
    'excluded': ['sub-30', 'sub-31', 'sub-35', 'sub-42'],

    # Montage to use
    'montage': 'standard_1005',
    # High pass filter cutoff
    'hpfilter': 0.1,
    # Low pass filter cutoff
    'lpfilter': 100,
    # Filter to use
    'filtertype': 'fir',
    # Plot for visual inspection (in Ipython, change pyplot to QT5)
    'visualinspect': False,
    # Reference
    'ref': 'average',
    # ICA parameters
    # Decimation factor before running ICA
    'icadecim': 4,
    # Set to get same decomposition each run
    'random_state': 23,
    # How many components keep in PCA
    'n_components': None,
    # Reject trials exceeding this amplitude before ICA
    'erpreject': dict(eeg=500e-6),
    # Algorithm
    'icamethod': 'fastica',
    # Visually identified bad channels
    'badchannels': {'23': ['T7', 'F6'],
                    '24': ['F7', 'FC4', 'Fp1', 'AF4', 'FC3'],
                    '25': ['AF7', 'AF8'],  # None
                    '26': ['P9'],
                    '27': ['AF7', 'Fp2'],
                    '28': [],
                    '29': ['F5'],
                    '30': ['CP6', 'TP8', 'P10', 'CP5'],
                    '31': ['T8', 'FC5', 'TP8'],  # Exclude
                    '32': ['P9'],
                    '33': ['F6', 'FT8', 'AF7'],
                    '34': ['T7', 'Iz'],
                    '35': [],  # Exclude
                    '36': ['P9', 'FC3'],  # None
                    '37': ['AFz', 'AF8'],  # None
                    '38': ['P10', 'O2'],  # None
                    '39': ['P9'],  # None
                    '40': ['Fpz', 'Fp2'],  # None
                    '41': ['TP7'],  # None
                    '42': ['CP2', 'P3', 'AF8', 'T7', 'CP5'],  # Exclude
                    '43': ['FC2', 'FC5'],  # None
                    '44': ['FC2', 'AF8'],  # None
                    '45': ['T7'],  # None
                    '46': [],  # None
                    '47': ['T8'],  # None
                    '48': [],  # None,
                    '49': [],  # None
                    '50': [],  # None
                    '51': ['AF3'],  # None
                    '52': ['P10'],  # None
                    '53': ['P9', 'AF7'],  # None
                    '54': ['AF3', 'Fp1'],  # None
                    '55': ['Fp2', 'F6', 'AF8'],  # None
                    '56': ['P2', 'P10', 'AF7', 'Iz', 'P10'],
                    '57': ['Oz', 'O1', 'TP8']},
    # Visually identified bad ICAS
    'badica': {'23': [0, 4],
               '24': [1, 2,  3, 4],
               '25': [2, 10],
               '26': [9],
               '27': [0, 6, 8],
               '28': [0, 6],
               '29': [0],
               '30': [0],
               '31': [0, 3, 4],
               '32': [0, 31, 34],
               '33': [0, 2, 3, 5],
               '34': [0, 45],
               '35': [0, 1, 2, 3, 4],
               '36': [0, 2],
               '37': [0, 6],
               '38': [2],
               '39': [0, 1],
               '40': [0, 14, 5, 6],
               '41': [3],
               '42': [0, 2, 4, 9, 10],
               '43': [0],
               '44': [2],
               '45': [0, 1],
               '46': [1, 15],
               '47': [0, 3, 4],
               '48': [2, 7],
               '49': [0, 1],
               '50': [0, 8],
               '51': [0, 2],
               '52': [0],
               '53': [0, 6, 7],
               '54': [0, 2, 3, 5],
               '55': [0, 2],
               '56': [0, 2, 3],
               '57': [0, 5, 8, 9]},
    'badica_shock': {'23': [23, 27, 31, 37, 38, 47, 51, 54, 57, 63],
                     '24': [5, 10, 13, 17, 22, 25, 31, 32, 33, 34, 39, 40, 42,
                            49, 41, 53, 54, 59, 60, 61, 62, 63],
                     '25': [11, 13, 26, 28, 29, 30, 31, 32, 33, 34, 37, 38,
                            41, 45, 58, 60, 62],
                     '26': [5, 6, 7, 16,  27, 28, 56, 60, 61, 64],
                     '27': [1, 14, 15, 16, 29, 33, 45, 53, 55, 56, 58, 59, 64],
                     '28': [4, 5, 12, 19, 25, 32, 33, 37, 40, 46, 47, 49],
                     '29': [3, 16, 29, 32],
                     '30': [27, 33, 35, 36, 40, 42, 45, 62, 63, 64],
                     '31': [],
                     '32': [6, 13, 14, 20, 23, 35, 36, 39, 40, 46, 56, 57,
                            61, 64],
                     '33': [3, 4, 6, 10, 11, 27, 28, 29, 30, 38, 43, 52, 53, 57, 58],
                     '34': [1, 5, 8, 11, 1, 32, 33, 34, 35, 36, 38, 39, 40, 43,
                            44, 56, 57, 58],
                     '35': [],
                     '36': [31, 40, 44, 48, 53, 56, 58, 63],
                     '37': [6, 10, 12, 21, 31, 38, 39, 41, 46, 54, 60],
                     '38': [18, 19, 21, 27, 31, 38, 49, 57, 59, 60, 61, 62],
                     '39': [10, 11, 12, 18, 19, 21, 25, 26, 32, 35, 38,
                            46, 52, 56, 59, 62, 63],
                     '40': [3, 6, 9, 19, 32, 34, 36, 38, 53, 57, 64],
                     '41': [10, 11, 12, 17, 29, 30, 34, 36, 37, 41, 42, 46,
                            53, 56],
                     '42': [0, 5, 9, 13, 22, 23, 24, 33, 35, 38, 39, 40,
                            42, 44, 48, 49, 50, 51, 52, 53, 54, 56, 57,
                            58, 59, 62, 63],
                     '43': [5, 6, 10, 11, 12, 16, 17, 18, 19, 29, 32, 35,
                            36, 37, 38, 41, 46, 53],
                     '44': [4, 5, 21, 48, 53, 56, 59, 60, 62, 64],
                     '45': [19, 23, 27, 32, 35, 39, 44, 54, 56, 59, 60, 61,
                            62, 63, 64],
                     '46': [3, 6, 18, 23, 32, 34, 39, 42, 43, 51, 52, 57, 63],
                     '47': [13, 24, 26, 30, 34, 35, 48, 52, 55, 61, 63, 64],
                     '48': [6, 18, 29, 34, 35, 37, 39, 42, 43, 47, 52, 57, 58, ],
                     '49': [10, 15, 19, 26, 38, 40, 44, 49, 53, 55, 60, 61,
                            62, 63, 65],
                     '50': [4, 24, 27, 34, 36, 47, 51, 56, 64],
                     '51': [0, 7, 9, 16, 21, 37, 41, 48, 57, 58],
                     '52': [8, 9, 20, 39, 45, 50, 56, 63, 64],
                     '53': [17, 18, 23, 28, 31, 54, 58, 65],
                     '54': [10, 18, 25, 31, 38, 39, 45, 46, 52, 53,
                            61, 62, 63, 64],
                     '55': [10, 30, 35, 46, 47, 55, 58, 60, 65],
                     '56': [5, 11, 12, 14, 18, 19, 21, 22, 28, 33, 34, 36,
                            41, 45, 50, 55, 60, 61],
                     '57': [1, 2, 3, 12, 27, 28, 35, 40, 50, 56, 59, 62, 63]},
}

part = [p for p in part if p not in param['excluded']]
part.sort()

# Output dir
outdir = opj(outpath)


for p in part:
    import warnings
    warnings.simplefilter('ignore')

    ###############################
    # Initialise
    ##############################
    print('Processing participant '
          + p)

    # _______________________________________________________
    # Make fslevel part dir
    pdir = opj(outdir, p, 'eeg')
    if not os.path.exists(pdir):
        os.mkdir(pdir)

    # _______________________________________________________
    # Initialise MNE report
    report = Report(verbose=False, subject=p,
                    title='EEG report for part ' + p)

    # report.add_htmls_to_section(
    #     htmls=part.comments[p], captions='Comments', section='Comments')
    report.add_htmls_to_section(
        htmls=pprint.pformat(param),
        captions='Parameters',
        section='Parameters')

    # ______________________________________________________
    # Load EEG file
    f = layout.get(subject=p[-2:], extension='bdf', return_type='filename')[0]
    raw = mne.io.read_raw_bdf(f, verbose=False,
                              eog=param['eogchan'],
                              exclude=param['dropchan'],
                              preload=True)

    # Rename external channels
    raw.rename_channels(param['renamechan'])

    events = pd.read_csv(layout.get(subject=p[-2:], extension='tsv',
                                    suffix='events',
                                    return_type='filename')[0], sep='\t')

    # For part with sampling at 2048 Hz, downsample
    if raw.info['sfreq'] == 2048:
        print(p)
        print('Resampling data to 1024 Hz')
        raw.resample(1024)
        events['sample'] = [e[0] for e in mne.find_events(raw)]
    # ______________________________________________________
    # Get events

    # Keep only rows for cues
    events_c = events[events['trial_type'].notna()]  # Cues
    events_s = events[events.trigger_info == 'shock']  # Shcok

    # Get events count
    events_count = events_c.trial_type.value_counts()

    # ReCalculate duration between events to double check
    events_c['time_from_prev2'] = np.insert((np.diff(events_c['sample'].copy()
                                                     / raw.info['sfreq'])),
                                            0, 0)

    events_c.to_csv(opj(pdir, p + '_task-fearcond_events.csv'))
    pd.DataFrame(events_count).to_csv(opj(pdir,
                                          p
                                          + '_task-fearcond_eventscount.csv'))

    # ______________________________________________________
    # Load and apply montage

    raw = raw.set_montage(param['montage'])
    raw.load_data()  # Load in RAM

    # ________________________________________________________________________
    # Remove bad channels
    if param['visualinspect']:
        raw.plot(
            n_channels=raw.info['nchan'],
            scalings=dict(eeg=0.00020),
            block=True)

    raw.info['bads'] = param['badchannels'][p[-2:]]

    # Plot sensor positions and add to report
    plt_sens = raw.plot_sensors(show_names=True, show=False)
    report.add_figs_to_section(
        plt_sens,
        captions='Sensor positions (bad in red)',
        section='Preprocessing')

    # _______________________________________________________________________
    # Bandpass filter
    raw_ica = raw.copy()  # Create a copy  to use different filter for ICA

    raw = raw.filter(
        param['hpfilter'],
        None,
        method=param['filtertype'],
        verbose=True)

    # ______________________________________________________________________
    # Plot filtered spectrum
    plt_psdf = raw.plot_psd(
        area_mode='range', tmax=10.0, average=False, show=False)
    report.add_figs_to_section(
        plt_psdf, captions='Filtered spectrum', section='Preprocessing')

    # From here, split cues and shocks trials
    raw_cues = raw.copy()
    raw_shocks = raw.copy()
    raw = None

    # ________________________________________________________________________
    # Clean with ICA

    # Make epochs around trial for ICA
    events_c['empty'] = 0
    events_c['triallabel'] = ['trial_' + str(i) for i in range(1, 469)]
    events_c['trialnum'] = range(1, 469)
    events_array_cues = np.asarray(events_c[['sample', 'empty', 'trialnum']])

    events_s['empty'] = 0
    events_s['triallabel'] = ['trial_' + str(i) for i in range(1, 55)]
    events_s['trialnum'] = range(1, 55)
    events_array_shocks = np.asarray(events_s[['sample', 'empty', 'trialnum']])

    alltrialsid_cues = {}
    for idx, name in enumerate(list(events_c['triallabel'])):
        alltrialsid_cues[name] = int(idx + 1)

    alltrialsid_shocks = {}
    for idx, name in enumerate(list(events_s['triallabel'])):
        alltrialsid_shocks[name] = int(idx + 1)

    # Low pass more agressively for ICA
    raw_ica = raw_ica.filter(l_freq=1, h_freq=100)
    epochs_ICA_cues = mne.Epochs(
        raw_ica,
        events=events_array_cues,
        event_id=alltrialsid_cues,
        tmin=-0.5,
        baseline=None,
        tmax=1,
        preload=True,
        reject=param['erpreject'],
        verbose=False)

    epochs_ICA_shock = mne.Epochs(
        raw_ica,
        events=events_array_shocks,
        event_id=alltrialsid_shocks,
        tmin=-0.5,
        baseline=None,
        tmax=0.5,
        reject=param['erpreject'],
        preload=True,
        verbose=False)

    print('Processing ICA for part ' + p + '. This may take some time.')
    ica = ICA(n_components=param['n_components'],
              method=param['icamethod'],
              random_state=param['random_state'])
    ica.fit(epochs_ICA_cues)

    # Add topo figures to report
    plt_icacomp = ica.plot_components(show=False, res=25)
    for l in range(len(plt_icacomp)):
        report.add_figs_to_section(
            plt_icacomp[l], captions='ICA', section='Artifacts')

    # Get manually identified bad ICA
    icatoremove = param['badica'][p[-2:]]

    # Identify which ICA correlate with eye blinks
    chaneog = 'VEOGL'
    eog_averagev = create_eog_epochs(raw_ica, ch_name=chaneog,
                                     verbose=False).average()
    # Find EOG ICA via correlation
    eog_epochsv = create_eog_epochs(
        raw_ica, ch_name=chaneog, verbose=False)  # get single EOG trials
    eog_indsv, scoresr = ica.find_bads_eog(
        eog_epochsv, ch_name=chaneog, verbose=False)  # find correlation

    fig = ica.plot_scores(scoresr, exclude=eog_indsv, show=False)
    report.add_figs_to_section(fig, captions='Correlation with EOG',
                               section='Artifact')

    # Get ICA identified in visual inspection
    figs = list()

    # Plot removed ICA and add to report
    ica.exclude = icatoremove
    figs.append(ica.plot_sources(eog_averagev,
                                 show=False,
                                 title='ICA removed on eog epochs'))

    report.add_figs_to_section(figs, section='ICA',
                               captions='Removed components '
                               + 'highlighted')

    report.add_htmls_to_section(
        htmls="IDX of removed ICA: " + str(icatoremove),
        captions='ICA-Removed',
        section='Artifacts')

    report.add_htmls_to_section(htmls="Number of removed ICA: "
                                + str(len(icatoremove)), captions="""ICA-
                                Removed""", section='Artifacts')

    # Loop all ICA and make diagnostic plots for report
    figs = list()
    capts = list()

    f = ica.plot_properties(epochs_ICA_cues,
                            picks='all',
                            psd_args={'fmax': 35.},
                            show=False)

    for ical in range(len(ica._ica_names)):

        figs.append(f[ical])
        capts.append(ica._ica_names[ical])

        ica.exclude = [ical]
        figs.append(ica.plot_sources(eog_averagev,
                                     show=False))
        plt.close("all")

    f = None
    report.add_slider_to_section(figs, captions=None,
                                 section='ICA-FULL')

    # Remove components manually identified
    ica.exclude = icatoremove

    # Apply ICA
    ica.apply(raw_cues)

    # Shocks ICA cleaning
    # Make new report just for shocks ICA
    report_shocks = Report(verbose=False, subject=p,
                           title='EEG shocks report for part ' + p)

    # Fit ICA on shocks epochs
    ica = ICA(n_components=param['n_components'],
              method=param['icamethod'],
              random_state=param['random_state'])
    ica.fit(epochs_ICA_shock)

    # Add topo figures to report
    plt_icacomp = ica.plot_components(show=False, res=25)
    for l in range(len(plt_icacomp)):
        report_shocks.add_figs_to_section(
            plt_icacomp[l], captions='ICA', section='Artifacts')

    figs = list()

    f = ica.plot_properties(epochs_ICA_shock,
                            picks=np.arange(ica.n_components_),
                            psd_args={'fmax': 100.},
                            show=False)

    # Apply ICA shocks
    ica_sources = ica.get_sources(epochs_ICA_shock)
    from mne.time_frequency import tfr_morlet

    ica_sources_tf = tfr_morlet(ica_sources,
                                freqs=np.arange(31, 91),
                                n_cycles=10,
                                return_itc=False,
                                use_fft=True,
                                picks=np.arange(
                                    ica_sources.get_data().shape[1]),
                                decim=4,
                                n_jobs=-1,
                                average=False)

    # Z score each component within each frequency
    data_z = ica_sources_tf.data.copy()
    for tr in range(ica_sources_tf.data.shape[0]):
        for ic in range(ica_sources_tf.data.shape[1]):
            for freq in range(ica_sources_tf.data.shape[2]):
                data_z[tr, ic, freq, :] = zscore(data_z[tr, ic, freq, :])

    ica_sources_tf.data = data_z

    figs1, figs2, figs3, figs4 = list(), list(), list(), list()
    for ical in range(len(ica._ica_names)):

        figs1.append(f[ical])

        ica.exclude = [ical]

        figs2.append(ica.plot_components(ical, show=False))
        figs3.append(ica_sources_tf.average().plot(ical, show=False)[0])
        figs4.append(ica_sources.plot_image(ical, show=False)[0])

        plt.close("all")

    report_shocks.add_slider_to_section(figs1, captions=None,
                                        section='ICA-shocks')
    report_shocks.add_slider_to_section(figs2, captions=None,
                                        section='ICA-shocks')

    report_shocks.add_slider_to_section(figs3, captions=None,
                                        section='ICA-shocks')
    report_shocks.add_slider_to_section(figs4, captions=None,
                                        section='ICA-shocks')

    report_shocks.add_figs_to_section(ica.plot_sources(epochs_ICA_shock, show=False,
                                                       picks=np.arange(0, 20),
                                                       start=0, stop=60),
                                      section='ICA-shocks', captions=['ICA_0-20'])
    report_shocks.add_figs_to_section(ica.plot_sources(epochs_ICA_shock, show=False,
                                                       picks=np.arange(20, 40),
                                                       start=0, stop=60),
                                      section='ICA-shocks', captions=['ICA_21-40'])
    report_shocks.add_figs_to_section(ica.plot_sources(epochs_ICA_shock, show=False,
                                                       picks=np.arange(
                                                           40, ica.n_components_),
                                                       start=0, stop=60),
                                      section='ICA-shocks', captions=['ICA_40_end'])

    # Remove components manually identified
    ica.exclude = param['badica_shock'][p[-2:]]
    ica.apply(raw_shocks)

    # ______________________________________________________________________
    # Re-reference data
    raw_cues, _ = mne.set_eeg_reference(
        raw_cues, param['ref'], projection=False)
    raw_shocks, _ = mne.set_eeg_reference(
        raw_shocks, param['ref'], projection=False)

    # Interpolate channels
    if raw_cues.info['bads']:
        raw_cues.interpolate_bads(reset_bads=True)
        raw_shocks.interpolate_bads(reset_bads=True)
    # ______________________________________________________________________
    # Save cleaned data
    raw_cues.save(opj(pdir, p + '_task-fearcond_cleanedeeg_cues_raw.fif'),
                  overwrite=True)
    raw_shocks.save(opj(pdir, p + '_task-fearcond_cleanedeeg_shocks_raw.fif'),
                    overwrite=True)
    raw_cues = None
    raw_shocks = None
    raw_ica = None
    #  _____________________________________________________________________
    report.save(opj(pdir, p + '_task-fearcond_importclean_report_cues.html'),
                open_browser=False, overwrite=True)

    report_shocks.save(opj(pdir, p + '_task-fearcond_importclean_report_shocks.html'),
                       open_browser=False, overwrite=True)

# For manuscript
stats_dict = {}

# Number of bad channels
nbc = []
for p, bc in param['badchannels'].items():
    if 'sub-' + p not in param['excluded']:
        nbc.append(len(bc))

nbic = []
for p, bic in param['badica'].items():
    if 'sub-' + p not in param['excluded']:
        nbic.append(len(bic))

nbics = []
for p, bic in param['badica_shock'].items():
    if 'sub-' + p not in param['excluded']:
        nbics.append(len(bic))


stats_dict = pd.DataFrame()
stats_dict['sub'] = part
stats_dict['n_bad_ica'] = nbic
stats_dict['n_bad_channels'] = nbc
stats_dict['n_bad_ica_shock'] = nbics
stats_dict.describe().to_csv(opj(outpath,
                                 'task-fearcond_importclean_stats.csv'))
