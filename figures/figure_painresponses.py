'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''

from os.path import join as opj
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from bids import BIDSLayout
import ptitprince as pt
from scipy.io import loadmat
from statsmodels.stats.multitest import multipletests
###############################
# Parameters
###############################

inpath = '/media/mp/mpx6/2023_painlerarnign_validate/source'
outpathall = '/media/mp/mpx6/2023_painlerarnign_validate/derivatives'


layout = BIDSLayout(inpath)
part = ['sub-' + s for s in layout.get_subject()]


param = {
    # Font sizez in plot
    'titlefontsize': 12,
    'labelfontsize': 12,
    'ticksfontsize': 11,
    'legendfontsize': 10,
}

# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Liberation Sans Narrow'


outfigpath = opj(outpathall, 'figures/pain_responses')
outpath = opj(outpathall, 'statistics/pain_responses')
if not os.path.exists(outpath):
    os.mkdir(outpath)
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)

#  ################################################################
# Figure relationship matrix
#################################################################

t_out = np.load(opj(outpath, 'slopes_tvals.npy'))
p_out = np.load(opj(outpath, 'slopes_pvals.npy'))
var_names = list(np.load(opj(outpath, 'slopes_varnames.npy')))


fig, ax = plt.subplots(figsize=(4, 4))


# Create masks for upper diagnoal and significance
sqsize = int(np.sqrt(t_out.shape))
mask = np.triu(np.ones_like(t_out.reshape((sqsize, sqsize))))

# Get unique pvals
iv_mask = ~mask.astype(bool)
pvals_unique = p_out.copy().reshape((sqsize, sqsize))[iv_mask]

# Get significance (with correction)
p_corrected = multipletests(pvals_unique, alpha=0.05, method='holm')[1]

# Put corrected values in matrix
p_out_corrected = p_out.reshape((sqsize, sqsize))
p_out_corrected[iv_mask] = p_corrected


sig = np.where(p_out_corrected < 0.05, 0, 1)
mask_sig = np.where((sig + mask) == 0, 0, 1)
mask_notsig = np.where(mask_sig == 0, 1, 0) + mask
mask_diag = np.ones(sqsize**2).reshape(sqsize, sqsize)
mask_diag[np.diag_indices(sqsize)] = 0


t_out.reshape((sqsize, sqsize))[np.diag_indices(sqsize)] = 3
t_out_annot = np.round(t_out, 3).astype(str).reshape((sqsize, sqsize))


im1 = sns.heatmap(t_out.reshape((sqsize, sqsize)), square=True, vmin=0, vmax=6,
                  annot=True,
                  cmap='viridis', mask=mask, axes=ax,
                  annot_kws=dict(fontsize=param['ticksfontsize']),
                  cbar_kws={"shrink": .75})

im2 = sns.heatmap(t_out.reshape((sqsize, sqsize)), square=True, vmin=0, vmax=6,
                  cmap='Greys', mask=mask_notsig, axes=ax, annot=True, cbar=False,
                  annot_kws=dict(fontsize=param['ticksfontsize']))
im = sns.heatmap(mask_diag, square=True, vmin=-10, vmax=10, center=0,
                 cmap='seismic', mask=mask_diag, axes=ax, annot=False,
                 cbar=False)

# Correct axis
left, right = ax.get_xlim()
ax.set_xlim(left, right-1)
ax.tick_params(bottom=False, left=False)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=param['ticksfontsize'], bottom=False, left=False)
cax.set_ylabel('Multilevel t-value for slope ',
               fontsize=param['labelfontsize'], labelpad=10)
var_names_y = list(var_names)
var_names_y[0] = ''
ax.set_yticklabels(var_names_y,
                   rotation=0, fontsize=param['labelfontsize'], va="center")
ax.set_xticklabels(var_names, fontsize=param['labelfontsize'])

fig.savefig(opj(outfigpath, 'pain_corrmatrix.svg'),
            dpi=600, bbox_inches='tight')
