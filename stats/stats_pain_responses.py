####################################################################
# Relationship between N2/P2 and computational estimates
#####################################################################

from mne.viz import plot_topomap
import mne
from os.path import join as opj
import pandas as pd
import numpy as np
import os
import scipy
import scipy.stats
from scipy.stats import zscore
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import ptitprince as pt
from tqdm import tqdm

# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Liberation Sans Narrow'

####################################################################
# Relationship between GAMMA multitaper computational estimates
#####################################################################

inpath = '/media/mp/mpx6/2023_painlerarnign_validate/source'
outpathall = '/media/mp/mpx6/2023_painlerarnign_validate/derivatives'


outfigpath = opj(outpathall, 'figures/pain_responses')
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)
outpath = opj(outpathall, 'statistics/pain_responses')
if not os.path.exists(outpath):
    os.mkdir(outpath)

# Load model data
mod_data = pd.read_csv(opj(outpathall, 'task-fearcond_alldata.csv'))

# Keep only reinforced trials
mod_data = mod_data[mod_data.trial_type == 'CS+S']

# Get shock prediction error
mod_data['da'] = 1 - mod_data['vhat']


# Empty frames to collect data
epos_tf, epos_erp, betas, values, allframes = [], [], [], [], []
good_eeg, good_null = [], []
epos_cues, epos_cuestf = [], []
excluded = ['sub-31', 'sub-35', 'sub-30', 'sub-42']
part = [s for s in mod_data['sub'].unique() if s not in excluded]

# Loop participants
part.sort()
for p in tqdm(part):
    print(p)
    subdat = mod_data[mod_data['sub'] == p].reset_index()

    # Load single trials for erps/tfr to shock/cues
    epo_tf = mne.time_frequency.read_tfrs(opj(outpathall,  p, 'eeg', 'tfr',
                                              p + '_task-fearcond_shock_epochs-tfr.h5'))[0]

    # ERp to shocks
    epo_erp = mne.read_epochs(opj(outpathall,  p, 'eeg', 'erps',
                                  p + '_task-fearcond_shock_singletrials-epo.fif'))

    # Add subject_id to DFs
    epo_erp.metadata['subject_id'] = p
    epo_tf.metadata['subject_id'] = p

    epo_erp.metadata = pd.concat(
        [epo_erp.metadata, subdat], 1).drop_duplicates()
    epo_tf.metadata = pd.concat([epo_tf.metadata, subdat], 1).drop_duplicates()

    epo_erp.metadata = epo_erp.metadata.loc[:,
                                            ~epo_erp.metadata.columns.duplicated()]
    epo_tf.metadata = epo_tf.metadata.loc[:,
                                          ~epo_tf.metadata.columns.duplicated()]

    # Apply baseline for TF trials
    epo_tf = epo_tf.apply_baseline((-0.5, -0.2), mode='logratio')

    # Flag bad trials in EEG
    print(part)
    epo_erp.metadata['goodtrials'] = np.where(
        epo_erp.metadata.badtrial == 0, 1, 0)

    # Get good trials
    subdat['good_eeg'] = np.asarray(epo_erp.metadata.goodtrials).astype(int)
    part
    print(np.sum(subdat['good_eeg']))

    # Get good trials for each participant
    good_eeg.append(np.asarray(epo_erp.metadata.goodtrials.astype(bool)))

    # For behavioural measures, no bad trials
    good_null.append(np.ones(54))

    # Z score data
    subdat['da_z'] = zscore(subdat['da'])
    subdat['vhat_z'] = zscore(subdat['vhat'])
    subdat['nfr_auc_z'] = zscore(np.log(subdat['nfr_auc'] + 1))
    subdat['ratings_z'] = zscore(subdat['ratings'])

    # Collect all
    allframes.append(subdat)
    epos_tf.append(epo_tf)
    epos_erp.append(epo_erp)


# Collect mean values in the gamma band (0.12-0.25 s, 60-85 Hz)
pick = epos_erp[0].ch_names.index('Cz')
values_tf, values_tf_z = [], []
for idx, p in enumerate(part):
    epo = epos_tf[idx].copy()
    out = epo.copy().crop(tmin=0.12, tmax=0.25, fmin=60, fmax=85).data
    dat = np.average(out[:, pick, :, :], axis=(1, 2))
    values_tf.append(dat)
    values_tf_z.append(zscore(dat))

# Collect n2/p2 peak difference
pick = epos_erp[0].ch_names.index('CPz')
values_n2p2 = []
values_n2p2_z = []
for idx, p in enumerate(part):
    epo = epos_erp[idx].copy()
    n2min = np.min(epo.copy().crop(tmin=0.05,
                                   tmax=0.2).get_data()[:, pick, :],
                   axis=1)
    p2peak = np.max(epo.copy().crop(tmin=0.2,
                                    tmax=0.5).get_data()[:, pick, :],
                    axis=1)
    n2p2 = p2peak-n2min
    values_n2p2.append(n2p2)
    values_n2p2_z.append(zscore(n2p2))

# Zscore everything
all_dat = pd.concat(allframes)

all_dat['n2p2'] = np.hstack(values_n2p2)
all_dat['n2p2_z'] = np.hstack(values_n2p2_z)
all_dat['gamma'] = np.hstack(values_tf)
all_dat['gamma_z'] = np.hstack(values_tf_z)


all_dat.to_csv(opj(outpath, 'pain_responses_medata.csv'))


# Test the relationship between all variables
all_betas = []
all_corrcoef = []
for idx, p in enumerate(part):
    # Extract all data for this part
    subdat = allframes[idx]
    n2p2 = values_n2p2[idx]
    tf = values_tf[idx]
    ratings = np.asarray(subdat['ratings'])
    nfr_auc = np.asarray(subdat['nfr_auc'])
    da = np.asarray(subdat['da'])
    vhat = np.asarray(subdat['vhat_z'])
    betas = []
    corrcoef = []

    var_list = [n2p2, tf, ratings, nfr_auc, da]
    good_trials = [good_eeg[idx], good_eeg[idx], good_null[idx],
                   good_null[idx], good_null[idx]]
    var_names = ('N2-P2\namplitude', '$\gamma$ power', 'Pain\nrating',
                 'NFR', 'Prediction\nError')

    for idx1, var1 in enumerate(var_list):
        for idx2, var2 in enumerate(var_list):

            # Find good trials in both variables
            g1 = good_trials[idx1]
            g2 = good_trials[idx2]
            good_all = np.where((g1.astype(int) + g2.astype(int)) == 2)

            # Get the data
            var1_corr = var1.copy()[good_all]
            var2_corr = var2.copy()[good_all]

            # Regress
            beta = stats.linregress(zscore(var1_corr), zscore(var2_corr))[0]
            betas.append(beta)

    # Stack in a sub x correlation matrix
    all_betas.append(betas)

# T-test against 0
t_out, p_out = stats.ttest_1samp(np.stack(all_betas), 0)


np.unique(p_out)
# Save for plots
np.save(opj(outpath, 'slopes_tvals.npy'), t_out)
np.save(opj(outpath, 'slopes_pvals.npy'), p_out)
np.save(opj(outpath, 'slopes_varnames.npy'), var_names)


def get_meansd_slopes(slopes, ci_se=1.96, xrange=[-100, 100]):
    meanslope = np.mean(slopes)
    seslope = sem(slopes)
    lowslope = meanslope - ci_se*seslope
    highslope = meanslope + ci_se*seslope

    meandata = meanslope*np.arange(xrange[0], xrange[1])
    lowdata = lowslope*np.arange(xrange[0], xrange[1])
    highdata = highslope*np.arange(xrange[0], xrange[1])

    return meandata, lowdata, highdata, np.arange(xrange[0], xrange[1])


current_palette = sns.color_palette('colorblind', 6)
labels = ['N2-P2 amplitude', '$\gamma$ power', 'Pain rating',
          'NFR']
for slidx in range(4):
    colp = current_palette[slidx]
    slopes = np.stack(all_betas)[:, 20+slidx]
    fig, ax1 = plt.subplots(figsize=(0.5, 2))
    # Add the intercept to the data

    pt.half_violinplot(y=slopes, inner=None,
                       width=0.6,
                       offset=0.17, cut=1, ax=ax1,
                       color=colp,
                       linewidth=1, alpha=0.6, zorder=19)
    sns.stripplot(y=slopes,
                  jitter=0.08, ax=ax1,
                  color=colp,
                  linewidth=1, alpha=0.6, zorder=1)
    sns.boxplot(y=slopes, whis=np.inf, linewidth=1, ax=ax1,
                width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                whiskerprops={'zorder': 10, 'alpha': 1},
                color=colp,
                medianprops={'zorder': 11, 'alpha': 0.5})
    ax1.set_ylabel('Slope (PE ~ ' + labels[slidx] + ')', fontsize=8)
    ax1.tick_params(axis='y', labelsize=7)
    ax1.set_xticks([], [])
    ax1.axhline(0, linestyle='--', color='k')
    plt.tight_layout()

    fig.savefig(opj(outfigpath, 'slopes_pe_' + labels[slidx].replace("$\gamma$ power", 'gamma') + '.svg'),
                dpi=600, bbox_inches='tight')
    from scipy.stats import sem

    mean, low, high, xdat = get_meansd_slopes(slopes)
    fig, ax = plt.subplots(figsize=(0.5, 2))
    plt.xlim((-3, 3))
    plt.ylim((-1, 1))
    sns.regplot(xdat, mean,  scatter=False, ci=68, line_kws={'linewidth': 1},
                color=current_palette[slidx], ax=ax, truncate=False)
    ax.fill_between(xdat, low, high, alpha=0.2, color=current_palette[slidx])

    plt.xlabel(labels[slidx] + ' (Z-scored)', fontsize=8)
    plt.ylabel('Prediction error (Z-scored)', fontsize=8)
    plt.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(opj(outfigpath, 'slopeslines_pe_' + labels[slidx].replace("$\gamma$ power", 'gamma') + '.svg'),
                dpi=600, bbox_inches='tight')

labels = ['N2-P2 amplitude', '$\gamma$ power', 'Pain rating',
          'NFR']
for slidx, var in zip(range(4), ['n2p2_z', 'gamma_z', 'ratings_z', 'nfr_auc_z']):
    colp = current_palette[slidx]

    fig, ax = plt.subplots(figsize=(1, 2))
    plt.xlim((-2.5, 2.5))
    plt.ylim((-1, 1))
    for idx, p in enumerate(part):
        subdat = all_dat[all_dat['sub'] == p]
        sns.regplot(var, 'da_z', data=subdat, scatter=False, ci=False,
                    color='gray', line_kws=dict(alpha=0.2), ax=ax, truncate=True)

    sns.regplot(var, 'da_z', data=all_dat, scatter=False, ci=95,
                color=current_palette[slidx], ax=ax, truncate=True)
    plt.xlabel(labels[slidx] + ' (Z-scored)', fontsize=22)
    plt.ylabel('Prediction error (Z-scored)', fontsize=22)
    plt.tick_params(labelsize=18)
    fig.savefig(opj(outfigpath, 'slopeslines2_pe_' + labels[slidx].replace("$\gamma$ power", 'gamma') + '.svg'),
                dpi=600, bbox_inches='tight')


# Gamma sync
avg_tfr = mne.grand_average([e.copy().average() for e in epos_tf])
avg_tfr.drop_channels(['M1', 'M2'])
avg_tfr.plot_topomap(fmin=60, fmax=85, tmin=0.12, tmax=0.25, cmap='viridis')


fig, ax = plt.subplots(figsize=(2, 2))
avg_tfr_low = avg_tfr.copy().crop(fmin=1, fmax=30).pick('Cz')
im = ax.imshow(np.squeeze(avg_tfr_low.data), aspect='auto',
               origin='lower', cmap='viridis')
ax.set_xlabel('Time (ms)',   fontdict={'fontsize': 9})
ax.set_ylabel('Frequency (Hz)',   fontdict={'fontsize': 9})
xticks = [np.argmin(np.abs(avg_tfr_low.times-t))
          for t in [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]]
ticklab = [-500, -250, 0, 250, 500, 750, 1000]
ax.set_xticks(xticks)
ax.set_xticklabels(ticklab)
yticks = [np.argmin(np.abs(avg_tfr_low.freqs-t))
          for t in [4, 10, 15, 20, 25, 30]]
ticklab = [4, 10, 15, 20, 25, 30]
ax.set_yticks(yticks)
ax.set_yticklabels(ticklab)
ax.tick_params(labelsize=8)
cb = fig.colorbar(im, ax=ax)
cb.ax.tick_params(labelsize=8)
cb.ax.set_ylabel('Power (log ratio)', fontsize=9, labelpad=6)
fig.savefig(opj(outfigpath, 'gamma_bottom.svg'), bbox_inches='tight', dpi=600)

fig, ax = plt.subplots(figsize=(2, 2))
avg_tfr_low = avg_tfr.copy().crop(fmin=31, fmax=100).pick('Cz')
im = ax.imshow(np.squeeze(avg_tfr_low.data), aspect='auto',
               origin='lower', cmap='viridis')
xticks = [np.argmin(np.abs(avg_tfr_low.times-t))
          for t in [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]]
ticklab = [-500, -250, 0, 250, 500, 750, 1000]
ax.set_xticks(xticks)
ax.set_xticklabels(ticklab)
ax.set_ylabel('Frequency (Hz)',   fontdict={'fontsize': 9})

yticks = [np.argmin(np.abs(avg_tfr_low.freqs-t))
          for t in [40, 50, 60, 70, 80, 90, 100]]
ticklab = [40, 50, 60, 70, 80, 90, 100]
ax.set_yticks(yticks)
ax.set_yticklabels(ticklab)
ax.set_xticks([])
ax.tick_params(labelsize=8)
cb = fig.colorbar(im, ax=ax)
cb.ax.tick_params(labelsize=8)


cb.ax.set_ylabel('Power (log ratio)', fontsize=9, labelpad=1)
ax.set_xlabel('Time (ms)',   fontdict={'fontsize': 9}, color='white')

fig.savefig(opj(outfigpath, 'gamma_top.svg'), bbox_inches='tight', dpi=600)


# N2P2 ERP
avg_erp = mne.grand_average(
    [e.copy().crop(tmin=-0.2, tmax=0.5).average() for e in epos_erp])
fig, line_axis = plt.subplots(figsize=(2, 4))
sub_avg = []
for s in range(len(part)):
    sub_avg.append(np.squeeze(epos_erp[s].copy().drop_bad(reject=dict(
        eeg=500e-6)).average().crop(tmin=-0.2, tmax=0.5).pick('CPz').data))
sub_avg = np.stack(sub_avg)
gavg = np.mean(sub_avg, axis=0)*1000000

# Get standard error
sem = scipy.stats.sem(sub_avg, axis=0)*1000000

line_axis.plot(avg_erp.times*1000, gavg)
line_axis.fill_between(avg_erp.times*1000,
                       gavg-sem, gavg+sem, alpha=0.3)

line_axis.hlines(0, xmin=line_axis.get_xlim()[0],
                 xmax=line_axis.get_xlim()[1],
                 linestyle="--",
                 colors="gray")
line_axis.vlines(0, ymin=line_axis.get_ylim()[0]/2,
                 ymax=line_axis.get_ylim()[1]/2,
                 linestyle="--",
                 colors="gray")

line_axis.set_xlabel('Time (ms)',
                     fontdict={'size': 9})
line_axis.set_ylabel('Amplitude (uV)',
                     fontdict={'size': 9})

line_axis.set_xticks(np.arange(-200, 600, 200))
# line_axis.set_xticklabels(np.arange(0, 900, 100))
line_axis.tick_params(axis='both', which='major',
                      labelsize=8,
                      length=5, width=1, direction='out', color='k')
line_axis.spines['right'].set_visible(False)
line_axis.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig(opj(outfigpath, 'n2p2_erp.svg'),
            dpi=600, bbox_inches='tight')


avg_erp = mne.grand_average([e.copy().crop(
    tmin=-0.2, tmax=0.5).drop_bad(reject=dict(eeg=500e-6)).average()for e in epos_erp])
avg_erp.drop_channels(['M1', 'M2'])
fig, ax = plt.subplots(figsize=(0.7, 0.7))


plot_topomap(avg_erp.data[:, np.argmin(np.abs(avg_erp.times-0.11))],
             avg_erp.info,
             show=False,
             mask_params=dict(markersize=8),
             outlines='head',
             extrapolate='head',
             # mask=mask[chankeep],
             axes=ax,
             sensors=False,
             contours=False)
ax.set_title('110 ms', fontsize=9, pad=0.1)

fig.savefig(opj(outfigpath, 'topo110ms.svg'), dpi=600, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(0.7, 0.7))


plot_topomap(avg_erp.data[:, np.argmin(np.abs(avg_erp.times-0.25))],
             avg_erp.info,
             show=False,
             mask_params=dict(markersize=8),
             outlines='head',
             extrapolate='head',
             # mask=mask[chankeep],
             axes=ax,
             sensors=False,
             contours=False)
ax.set_title('250 ms', fontsize=9, pad=0.1)


fig.savefig(opj(outfigpath, 'topo250ms.svg'), dpi=600, bbox_inches='tight')
