import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm
from matplotlib.colors import ListedColormap

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

   parser.add_argument('-dir', metavar='DIR', help='Result dir.')
   parser.add_argument('-dataset', metavar='DATASET', help='Dataset to group.')
   parser.add_argument('-batch', action='store_true', help='Gen groupedimage.')
   parser.add_argument('-ex', nargs='*', metavar='EXCLUDES', help='Exclude mnemonics.')

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


def mean_hist_join_plot_batch(data, size, xlim=None, ylim=None, xlabel='', ylabel='', save=None):
    fig, axs = plt.subplots(size[0], size[1], figsize=(8, 5 * size[0]))

    i = 0
    sb =['s', '^', 'd', 'o', '*']
    for d in data:
        x = d['zphot - zspec']
        y = d['zspec']
        std = d['std']

        if size[0] == 1:
            axs.set_ylabel(ylabel, fontsize=16)
            axs.tick_params(axis='y', which='both', labelleft=True, labelsize=14)
            axs.set_xlabel(xlabel, fontsize=16)
            axs.tick_params(axis='x', which='both', labelbottom=True, labelsize=14)

            axs.plot(x, y, f"-{sb[i]}", label=d['title'])
#            axs.plot([x,-std,std], [y,y,y], f"-{sb[i]}", label=d['title'])
            i = i + 1

        else:
            if d['valset'] == 'B':
                axs[0].set_ylabel(ylabel, fontsize=16)
                axs[0].tick_params(axis='y', which='both', labelleft=True, labelsize=14)
                axs[0].set_xlabel(xlabel, fontsize=16)
                axs[0].tick_params(axis='x', which='both', labelbottom=True, labelsize=14)

                axs[0].plot(x, y, f"-{sb[i]}", label=d['title'])
                #axs[0].plot([x, -std, std], [y,y,y], f"-{sb[i]}", label=d['title'])

            if d['valset'] == 'C':
                axs[1].set_ylabel(ylabel, fontsize=16)
                axs[1].tick_params(axis='y', which='both', labelleft=True, labelsize=14)
                axs[1].set_xlabel(xlabel, fontsize=16)
                axs[1].tick_params(axis='x', which='both', labelbottom=True, labelsize=14)

                axs[1].plot(x, y, f"-{sb[i]}", label=d['title'])
                #axs[1].plot([x, -std, std], [y,y,y], f"-{sb[i]}", label=d['title'])

            if d['valset'] == 'D':
                axs[2].set_ylabel(ylabel, fontsize=16)
                axs[2].tick_params(axis='y', which='both', labelleft=True, labelsize=14)
                axs[2].set_xlabel(xlabel, fontsize=16)
                axs[2].tick_params(axis='x', which='both', labelbottom=True, labelsize=14)

                axs[2].plot(x, y, f"-{sb[i]}", label=d['title'])
                #axs[2].plot([x, -std, std], [y,y,y], f"-{sb[i]}", label=d['title'])
                i = i + 1

    if size[0] == 1:
        plt.legend()
    else:
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

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

    gist_earth = cm.get_cmap('gist_earth', 500)
    gist_rainbow = cm.get_cmap('rainbow', 500)
    newcolors = gist_earth(np.linspace(0, 1, 500))
    rainbow = gist_rainbow(np.linspace(0, 1, 500))
    white = np.array([1, 1, 1, 1])
    #white = newcolors[-6:, :]
    red = rainbow[-180:, :]
    newcolors[:6, :] = white
    newcolors[-180:, :] = red
    cmap = ListedColormap(newcolors)

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap)

    ax.set_title(title, fontsize=24, color='blue')
    ax.set_xlabel(x_label, fontsize=22, color='blue')
    ax.set_ylabel(y_label, fontsize=22, color='blue')

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

    zspec = data['Real']
    zphot = data['Pred']
    residual = (zphot - zspec) / (1 + zspec)

    title = f"{mnemonic} | {_MAP_DATASET_NAMES[dataset]}"
    save = f"{mnemonic}_{dataset}"
    if valset:
        title = f"{title}: {valset}"
        save = f"{save}_{valset}"

    if use_heatmap:
        heatmap_plot(zspec, residual, title, 'Z-Spec', '(Z-Phot - Z-Spec) / 1 + Z-Spec', xlim=(0, 0.6), ylim=(-.2, .2), save=save)
    else:
        plot_jointd_sct_m(zspec, residual, title, 'Z-Spec', '(Z-Phot - Z-Spec) / (1 + Z-Spec)', xlim=(0, 0.6), ylim=(-.2, .2), s=15, save=save)


def mean_hist_plot_batch_report(dir, dataset_criteria, exclusions):
    preds_file_mask = f"{dir}/real_x_pred_*"
    preds_files = glob.glob(preds_file_mask)
    #preds_data = pd.DataFrame(columns=['title', 'zspec', 'zphot - zspec', 'std', 'valset'])
    preds_data = np.array([])
    fpreds_files = np.array([])

    for f in preds_files:
        file = f.split('/')[-1]
        file_info = file.split("_")
        mnemonic = file_info[3]
        dataset = file_info[5]
        if dataset == dataset_criteria:
            if exclusions == None:
                fpreds_files = np.append(fpreds_files, f)
            elif mnemonic not in exclusions:
                fpreds_files = np.append(fpreds_files, f)

    fpreds_files.sort()

    for pf in fpreds_files:
        file = pf.split('/')[-1]
        print(f"######## {file} ########")
        file_info = file.split("_")
        mnemonic = file_info[3]
        scaler = file_info[4]
        dataset = file_info[5]
        valset = None
        if len(file_info) == 14:
            valset = file_info[-1]

        data = pd.read_csv(pf)
        zspec = data['Real']
        zphot = data['Pred']
        diff = zphot - zspec
        m = diff.mean()
        std = diff.std()

        title = f"{mnemonic} | {_MAP_DATASET_NAMES[dataset]}"

        if valset:
            title = f"{title}: {valset}"

        p_data = {'title': title, 'zspec' : data['Real'].mean(), 'zphot - zspec': m, 'std': std, 'valset': valset}
        preds_data = np.append(preds_data, p_data)
        #preds_data = preds_data.append(p_data, ignore_index=True)


    save = f"means_hist_join_{dataset_criteria}"
    size = [3, 1]
    if dataset_criteria == 'sdss':
        size = [1, 1]

    mean_hist_join_plot_batch(
        preds_data, size,
        xlim=(-1.2, 1.2), ylim=(0, 5000),
        save=save,
        xlabel='zphot - zspec',
        ylabel='zspec'
    )



if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    if(args.batch):
        mean_hist_plot_batch_report(args.dir, args.dataset, args.ex)
    else:
        residual_plot_report(args.file, args.hm)

