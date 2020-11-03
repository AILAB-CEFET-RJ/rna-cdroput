import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter



_MAP_MNEMONIC_NAMES = {
    'xgb5': 'XGB05',
    'xgb05': 'XGB05',
    'xgb10': 'XGB10',
    'rna05': 'RNA05',
    'rna10': 'RNA10',
    'rnari': 'RNA-RI',
    'rnari10': 'RNA-RI10',
    'rnaad': 'RNA-AD',
    'rnariinv': 'RNA-RI-Inv',
    'rnaadinv': 'RNA-AD-Inv',
    'rnarriinv': 'RNA-RRI-Inv',
    'rnaradinv': 'RNA-RAD-Inv',
    'rnaradinv10': 'RNA-RAD-Inv10'
}

_MAP_DATASET_NAMES = {
    'teddy': 'COIN/Teddy',
    'happy': 'COIN/Happy',
    'sdss': 'SDSS'
}

def parser():
   parser = argparse.ArgumentParser(description='Reports module')
   parser.add_argument('-file', metavar='RES', help='Result file.')
   parser.add_argument('-hm', action='store_true', help='Use Heatmap.')

   return parser


def plot_jointd_sct_m(xdf, ydf, title, xlabel, ylabel, xlim=None, ylim=None, s=0.5, save=None):
    xdf.reset_index(drop=True, inplace=True)
    ydf.reset_index(drop=True, inplace=True)
    x = xdf
    y = ydf

    # definitions for the axes
    spacing = 0.02
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(12, 12), constrained_layout=True)

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.set_title(title, fontsize=24, color='blue')
    axScatter.set_xlabel(xlabel, fontsize=22, color='blue')
    axScatter.set_ylabel(ylabel, fontsize=22, color='blue')

    axScatter.tick_params(axis='both', labelsize=18)
    axHistx.tick_params(axis='both', labelsize=18)
    axHisty.tick_params(axis='both', labelsize=18)

    # no labels
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, c='blue', s=s, alpha=0.03)

    binwidth = 0.05
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (xymax / binwidth + 1) * binwidth

    if (xlim == None):
        xlim = (-lim, lim)

    if (ylim == None):
        ylim = (-lim, lim)

    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)

    xmax = np.max(x)
    xmin = np.min(x)
    ymax = np.max(y)
    ymin = np.min(y)
    sup_lim_x = (int(xmax / binwidth) + 1) * binwidth
    inf_lim_x = (int(xmin / binwidth) - 1) * binwidth
    sup_lim_y = (int(ymax / binwidth) + 1) * binwidth
    inf_lim_y = (int(ymin / binwidth) - 1) * binwidth

    # bins = np.arange(-lim, lim + binwidth, binwidth)
    bins_x = np.arange(inf_lim_x, sup_lim_x + binwidth, binwidth)
    bins_y = np.arange(inf_lim_y, sup_lim_y + binwidth, binwidth)

    # axHistx.hist(x, bins=bins, color='blue', histtype='stepfilled')
    # axHisty.hist(y, bins=bins, color='blue', histtype='stepfilled', orientation='horizontal')
    axHistx.hist(x, bins=bins_x, color='blue', histtype='stepfilled')
    axHisty.hist(y, bins=bins_y, color='blue', histtype='stepfilled', orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    if save:
        plt.savefig(save)

    plt.show()


def heatmap_plot(x, y, title, x_label, y_label, xlim=None, ylim=None, save=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=500)

    if (xlim == None and ylim == None):
        x_start = np.amin(x)
        y_start = np.amin(y)
        x_end = np.amax(x)
        y_end = np.amax(y)

        if x_start < y_start:
            y_start = x_start
        else:
            x_start = y_start

        if x_end > y_end:
            y_end = x_end
        else:
            x_end = y_end

        extent = [x_start, x_end, y_start, y_end]
    else:
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

    im = plt.imshow(heatmap.T, extent=extent, origin='lower')

    #paleta de cores pre determinada
    #im.set_cmap('nipy_spectral')
    #im.set_cmap('flag')
    im.set_cmap('gist_earth')
    #im.set_cmap('gist_earth_r')
    #im.set_cmap('terrain')

    ax.set_title(title, fontsize=24, color='blue')
    ax.set_xlabel(x_label, fontsize=22, color='blue')
    ax.set_ylabel(y_label, fontsize=22, color='blue')

    #legenda das cores
    #plt.colorbar()

    if save:
        plt.savefig(save)

    plt.show()


def residual_plot_report(result_file, use_heatmap):
    data = pd.read_csv(result_file)
    print(data.info())
    print(data.head())

    file = result_file.split('/')[-1]
    print(f"######## {file} ########")

    file_info = file.split("_")
    mnemonic = file_info[3]
    dataset = file_info[5]
    valset = None
    if len(file_info) == 14:
        valset = file_info[-1]

    zspec = data['Pred']
    zphot = data['Real']
    residual = (zphot - zspec) / (1 + zspec)

    title = f"{mnemonic} | {_MAP_DATASET_NAMES[dataset]}"
    save = f"{mnemonic}_{dataset}"
    if valset:
        title = f"{title}: {valset}"
        save = f"{save}_{valset}"

    if use_heatmap:
        heatmap_plot(zspec, residual, title, 'Z-Spec', '(Z-Phot - Z-Spec) / 1 + Z-Spec', save=save)
    else:
        plot_jointd_sct_m(zspec, residual, title, 'Z-Spec', '(Z-Phot - Z-Spec) / (1 + Z-Spec)', xlim=(0, 0.6), ylim=(-.2, .2), s=15, save=save)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    residual_plot_report(args.file, args.hm)

