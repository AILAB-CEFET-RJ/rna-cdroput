import argparse
import numpy as np
import pandas as pd
import glob

from datetime import timedelta
from modules.plot import heatmap_plot

import matplotlib.pyplot as plt

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
    'rnaradinv10': 'RNA-RAD-Inv10',
    'rnaird05': 'RNA-DRI-Inv05',
    'rnaird10': 'RNA-DRI-Inv10',
    'rnairi05': 'RNA-RI-Inv05',
    'rnairi10': 'RNA-RI-Inv10',
    'rnairid05': 'RNA-DRI-Inv05',
    'rnairid10': 'RNA-DRI-Inv10',
    'rnaedt10': 'RNA-EAD10'
}

_MAP_SCALER_NAMES = {
    'StandardScaler': 'Padrão',
    None: 'N.A.'
}

_MAP_DATASET_NAMES = {
    'teddy': 'COIN/Teddy',
    'happy': 'COIN/Happy',
    'sdss': 'SDSS'
}

def parser():
   parser = argparse.ArgumentParser(description='Reports module')
   parser.add_argument('-dir', metavar='DIR', help='Result file directory.')
   parser.add_argument('-std_perc', action='store_true', help='Report of standard deviations distances.')
   parser.add_argument('-img_hist', action='store_true', help='Plot of best model learning curves.')
   parser.add_argument('-hm', action='store_true', help='Plot results heatmap.')
   parser.add_argument('-metric', metavar='METRIC', help='Select metric to show in [-img_hist]. Default is loss.')
   parser.add_argument('-sc', metavar='SCALER', help='Scaler class used.')
   parser.add_argument('-run', metavar='RUN', type=int, help='Select run.')
   parser.add_argument('-f', metavar='FEATURES', help='Number of features.')
   parser.add_argument('-e', metavar='EPOCHS', help='Number of epochs.')
   parser.add_argument('-dataset', metavar='DS', help='Dataset to use [teddy|happy|kaggle|kaggle_bkp].')
   parser.add_argument('-valset', metavar='VS', help='Valset to use [B|C|D].')
   parser.add_argument('-model', metavar='MODEL', help='Model mnemonic.')
   parser.add_argument('-pf', metavar='PF', help='Preds file.')
   parser.add_argument('-gen_table', action='store_true', help='Generate table from out files.')
   parser.add_argument('-gen_trace_table', action='store_true', help='Generate trace metrics table from out files.')
   parser.add_argument('-dzm_bias', action='store_true', help='Generate delta z mean for bias.')
   parser.add_argument('-residual_plot', action='store_true', help='Generate residual plot.')
   parser.add_argument('-save', metavar='FILE', help='Save the image in a file.')

   return parser


def dzm_bias_report(files_dir):
    tbl = pd.DataFrame(columns=[
        'valset', 'method', 'dataset', 'bias'
    ])

    pred_file_mask = f"{files_dir}/real_*"
    pred_files = glob.glob(pred_file_mask)
    for file in pred_files:
        print(f"######## {file} ########")
        infos = file.split('/')[-1].split('_')

        valset = '-'
        if len(infos) == 14:
            valset = infos[13]

        pred_df = pd.read_csv(f"{file}")
        zspec = pred_df['Pred']
        zphot = pred_df['Real']
        bias = (zphot - zspec).mean()

        data = {
            'valset': valset, 'method': infos[3], 'dataset': infos[5], 'bias': bias
        }

        tbl = tbl.append([data], ignore_index=True)


    dataset = 'teddy'
    valset = 'C'
    #tbl = tbl[tbl.dataset == dataset]
    #tbl = tbl[tbl.valset == valset]
    tbl = tbl.sort_values(['valset', 'method', 'dataset'])
    print(tbl.info())
    print('=================================================================')
    print(tbl.to_csv(index=False))
    print('=================================================================')
    print(tbl.to_latex(index=False))

    fig, ax = plt.subplots()
    plt.barh(tbl.method, tbl.bias)
    ax.set_title(f"{_MAP_DATASET_NAMES[dataset]} | {valset}")
    #ax.set_title(f"{_MAP_DATASET_NAMES[dataset]}")
    ax.set_xlabel('Média Delta Z')
    ax.set_ylabel('Método')

    plt.show()




def gen_std_perc_report(files_dir):
    tbl = pd.DataFrame(columns=[
        'method', 'scaler', 'dataset',
        '0-1', '1-2', '2-3', '3-4', '4+'
    ])

    pred_file_mask = f"{files_dir}/real_*"
    pred_files = glob.glob(pred_file_mask)
    for file in pred_files:
        print(f"######## {file} ########")
        data = {
            'method': None, 'scaler': None, 'dataset': None, 'valset': None,
            '0-1': None, '1-2': None, '2-3': None, '3-4': None, '4+': None
        }
        infos = file.split('_')

        data['method'] = infos[5]
        data['scaler'] = infos[6]
        data['dataset'] = infos[7]

        if len(infos) == 16:
            data['valset'] = infos[15]
        else:
            data['valset'] = '-'

        pred_df = pd.read_csv(f"{file}")
        b = [0, 1, 2, 3, 4]
        result = percent_distance_std(pred_df['Pred'], pred_df['Real'], b)
        print(result)
        print(f"============================================================")
        format_std_perc_report(b, result)
        result = result * 100

        data['0-1'] = np.round(result[0], 2)
        data['1-2'] = np.round(result[1], 2)
        data['2-3'] = np.round(result[2], 2)
        data['3-4'] = np.round(result[3], 2)
        data['4+'] = np.round(result[4], 2)
        tbl = tbl.append([data], ignore_index=True)

        print(f"############################################################")

    tbl = tbl[tbl.dataset == "teddy"]
    tbl = tbl[tbl.valset == "D"]
    tbl = tbl.sort_values(['valset', 'method', 'dataset'])
    tbl.drop(columns=['dataset', 'valset', 'scaler'], axis=1, inplace=True)

    latex_std_perc_report(tbl)

def latex_std_perc_report(tbl):
    print(tbl.to_csv(index=False))


def format_std_perc_report(b , result):
    j = 0
    for i in range(len(b)):
        if i > 0:
            acc = result[j] * 100
            print(f"total entre {b[i - 1]} e {b[i]} std: {acc:.2f}%")
            j = j + 1

    if j != len(result):
        acc = result[-1] * 100
        print(f"total não classificado: {acc:.2f}%")


def percent_distance_std(pred, real, bins):
    bins = np.array(bins, dtype='float32')
    diff = np.power(real - pred, 2)
    std = diff.std(ddof=0)

    print(f"std = {std}")
    bins = bins * std

    interval = []

    for i in range(len(bins)):
        if i > 0:
            interval.append(pd.Interval(left=bins[i - 1], right=bins[i]))

    ii = pd.IntervalIndex(interval,
                          closed='right',
                          dtype='interval[float32]')

    cut = pd.cut(diff, bins=ii)

    s = pd.Series(cut).value_counts(dropna=False).sort_index()
    counts = s.to_numpy()
    total = counts.sum()
    print(s)

    result = np.array([])
    for c in counts:
        percent = c / total
        result = np.append(result, percent)

    return result


def get_best_model_file(model_files):
    model_map={}
    for m in model_files:
        mse = float(m.split('mse')[1].split('_')[1].split('.')[0])
        model_map[mse] = m

    models = {k: model_map[k] for k in sorted(model_map)} # sort by mse
    best = list(models)[0]

    return model_map[best]


def gen_img_hist_report(files_dir, mnemonic, scaler, dataset, metric, epochs, run, save=None):
    model_file_mask = f"{files_dir}/model_{mnemonic}_{dataset}_{scaler}*"
    if epochs:
        model_file_mask = f"{files_dir}/model_{mnemonic}_{dataset}_{scaler}_epochs_{epochs}*"

    model_files = glob.glob(model_file_mask)
    model_files.sort()

    if len(model_files) > 0:
        best_model_file = get_best_model_file(model_files)
        idx = best_model_file.split('run_')[1].split('_')[0]
        if run != None:
            idx = run
        best_hist_file_mask = f"{files_dir}/hist_{mnemonic}_{scaler}_{dataset}*_run_{idx}"
        if epochs:
            best_hist_file_mask = f"{files_dir}/hist_{mnemonic}_{scaler}_{dataset}_epochs_{epochs}*_run_{idx}"
        best_hist_file = glob.glob(best_hist_file_mask)[0]
        print(f"######## {best_hist_file.split('/')[-1]} ########")
    else:
        print(">>> MODEL MISS !!!")
        idx = 10
        best_hist_file = get_hist_model_miss_report(files_dir, mnemonic, scaler, dataset)

    hist = pd.read_csv(best_hist_file)
    print(hist.info())
    print(hist.head())
    print(f"Max MSE: {hist['mse'].max()}")
    print(f"Min MSE: {hist['mse'].min()}")

    if metric == None:
        metric = 'loss'

    fig, ax = plt.subplots()
    hist[[metric, f"val_{metric}"]].plot.line(ax=ax)
    ax.set_title(f"{_MAP_DATASET_NAMES[dataset]}: {mnemonic} | Scaler[{_MAP_SCALER_NAMES[scaler]}] | Rodada {idx}")
    ax.legend(["train_loss", "val_loss"])
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Erros')

    if save:
        plt.savefig(save)

    plt.show()


def get_hist_model_miss_report(files_dir, mnemonic, scaler, dataset):
    file_mask = f"{files_dir}/hist_{mnemonic}_{scaler}_{dataset}*"
    files = glob.glob(file_mask)
    files.sort()
    last_hist_file = files[-1]
    print(f"######## {last_hist_file.split('/')[-1]} ########")

    return last_hist_file


def gen_heatmap_report(preds_file, files_dir, mnemonic, scaler, dataset, valset, extent=None):
    if not preds_file:
        preds_file_mask = f"{files_dir}/real_x_pred_{mnemonic}_{scaler}_{dataset}*"
        if valset:
            preds_file_mask = f"{files_dir}/real_x_pred_{mnemonic}_{scaler}_{dataset}*{valset}"

        preds_files = glob.glob(preds_file_mask)

        preds_file = preds_files[0]
        print(f"######## {preds_file.split('/')[-1]} ########")

    data = pd.read_csv(preds_file)
    print(data.info())
    print(data.head())

    fig, ax = plt.subplots()
    x = data['Pred']
    y = data['Real']
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)

    if (extent == None):
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

    title = f"{_MAP_DATASET_NAMES[dataset]}: {mnemonic} | Scaler[{_MAP_SCALER_NAMES[scaler]}]"
    if valset:
        title = f"{_MAP_DATASET_NAMES[dataset]} {valset}: {mnemonic} | Scaler[{_MAP_SCALER_NAMES[scaler]}]"

    im = plt.imshow(heatmap.T, extent=extent, origin='lower')
    im.set_cmap('gist_heat_r')
    ax.set_title(title)
    ax.set_xlabel('Z-Spec')
    ax.set_ylabel('Z-Phot')
    plt.show()


def residual_plot_report(preds_file, files_dir, mnemonic, scaler, dataset, valset):
    if not preds_file:
        preds_file_mask = f"{files_dir}/real_x_pred_*"
        preds_files = glob.glob(preds_file_mask)
        for pf in preds_files:
            file = pf.split('/')[-1]
            print(f"######## {file} ########")
            file_info = file.split("_")
            mnemonic = file_info[3]
            scaler = file_info[4]
            dataset = file_info[5]
            valset = None
            if len(file_info) == 14:
                valset = file_info[-1]
            residual_plot_report(pf, None, mnemonic, scaler, dataset, valset)

    else:
        data = pd.read_csv(preds_file)
        print(data.info())
        print(data.head())

        zspec = data['Real']
        zphot = data['Pred']
        residual = (zphot - zspec) / (1 + zspec)

        title = f"{mnemonic} | {_MAP_DATASET_NAMES[dataset]}"
        save = f"{mnemonic}_{dataset}"
        if valset:
            title = f"{title}: {valset}"
            save = f"{save}_{valset}"
        #plot_jointd_sct_m(zspec, residual, title, 'Z-Spec', '(Z-Phot - Z-Spec) / (1 + Z-Spec)', xlim=(0, 0.6), ylim=(-.2, .2), s=15, save=save)
        heatmap_plot(zspec, residual, title, 'Z-Spec', '(Z-Phot - Z-Spec) / (1 + Z-Spec)', save=save)



def gen_table_report(dir):
    outs_file_mask = f"{dir}/out_*"
    outs_files = glob.glob(outs_file_mask)
    outs_files.sort()

    tbl = pd.DataFrame(columns=[
        'valset', 'method', 'dataset',
        'mse', 'mae', 'rmse', 'mad','r2', 'time'
    ])

    timereg = {}

    for out in outs_files:
        with open(out, 'r') as file:
            infos = out.split('/')[-1].split('_')

            for line in file:
                if '[CSV_t]' in line:
                    method = _MAP_MNEMONIC_NAMES[infos[1]]
                    dataset = _MAP_DATASET_NAMES[infos[3].split('.')[0]]
                    time = line.split('[CSV_t] ')[1]
                    timereg[f"{method}_{dataset}"] = fix_time(time)

                if '[CSV]' in line:
                    valset = '-'
                    if len(infos) > 4:
                        valset = infos[-1].split('.')[0]
                    d = line.split('[CSV] ')[1].replace("\n", '').split(',')
                    method = _MAP_MNEMONIC_NAMES[infos[1]]
                    dataset = _MAP_DATASET_NAMES[infos[3].split('.')[0]]
                    data = {
                        'method': method,
                        'dataset': dataset,
                        'valset': valset,
                        'mse': d[0], 'mae': d[1], 'rmse': d[2], 'mad': d[3],'r2': d[-1],
                        'time': timereg.get(f"{method}_{dataset}", None)
                    }
                    tbl = tbl.append([data], ignore_index=True)

    #tbl = tbl[tbl.dataset == "SDSS"]
    #tbl = tbl[tbl.valset == "D"]
    tbl = tbl.sort_values(['valset', 'dataset', 'method'])
    #tbl.drop(columns=['dataset', 'valset'], axis=1, inplace=True)
    #tbl.drop(columns=['mse', 'mae', 'rmse','r2'], axis=1, inplace=True)
    print(tbl.info())
    print('=================================================================')
    print(tbl.to_csv(index=False))
    print('=================================================================')
    print(tbl.to_latex(index=False).replace('±', '$\pm$'))


def fix_time(time):
    if ':' in time:
        times = time.split('±')

        d = times[0].split(' days, ')
        t0 = d[1].split(':')
        t1 = times[1].split(':')

        t = timedelta(days=int(d[0]), hours=int(t0[0]), minutes=int(t0[1]), seconds= float(t0[2]))
        tstd = timedelta(hours=int(t1[0]), minutes=int(t1[1]), seconds= float(t1[2]))

        t = round((t.total_seconds() / 1000) / 60)
        #tstd = (tstd.total_seconds() / 1000) / 60

        #print(t, tstd)

    else:
        times = time.split('±')

        t = round(float(times[0]) / 60)
        #tstd = round(float(times[1]) / 60)

        #print(t, tstd)

    return t


def gen_trace_table_report(dir):
    metrics = 'mses'
    outs_file_mask = f"{dir}/out_*"
    outs_files = glob.glob(outs_file_mask)
    outs_files.sort()

    tbl = pd.DataFrame(columns=[
        'valset', 'method', 'dataset',
        f"{metrics[:-1]}_0",f"{metrics[:-1]}_1",
        f"{metrics[:-1]}_2",f"{metrics[:-1]}_3",f"{metrics[:-1]}_4",
        f"{metrics[:-1]}_5",f"{metrics[:-1]}_6",f"{metrics[:-1]}_7",
        f"{metrics[:-1]}_8",f"{metrics[:-1]}_9"
    ])

    records = {}
    next_line=False

    for out in outs_files:
        with open(out, 'r') as file:
            infos = out.split('/')[-1].split('_')
            valset = '-'
            if len(infos) > 4:
                valset = infos[-1].split('.')[0]
            method = _MAP_MNEMONIC_NAMES[infos[1]]
            dataset = _MAP_DATASET_NAMES[infos[3].split('.')[0]]

            for line in file:
                if f"[CSV_{metrics}]" in line:
                    records[f"{method}_{dataset}_{valset}"] = line.replace("\n", '')
                    next_line = True
                elif next_line:
                    records[f"{method}_{dataset}_{valset}"] = records[f"{method}_{dataset}_{valset}"] + line.replace("\n", '')
                    next_line = False

            if not records:
                continue

    for k, v in records.items():
        infos = k.split('_')
        d = v.split(f"[CSV_{metrics}] ")[1] \
            .replace('   ', ' ')\
            .replace('  ', ' ')\
            .replace('[', '')\
            .replace(']', '')\
            .split(' ')

        data = {
            'method': infos[0],
            'dataset': infos[1],
            'valset': infos[2],
            f"{metrics[:-1]}_0": d[0],
            f"{metrics[:-1]}_1": d[1],
            f"{metrics[:-1]}_2": d[2],
            f"{metrics[:-1]}_3": d[3],
            f"{metrics[:-1]}_4": d[4],
            f"{metrics[:-1]}_5": d[5],
            f"{metrics[:-1]}_6": d[6],
            f"{metrics[:-1]}_7": d[7],
            f"{metrics[:-1]}_8": d[8],
            f"{metrics[:-1]}_9": d[9],
        }
        tbl = tbl.append([data], ignore_index=True)


    #tbl = tbl[tbl.dataset == "SDSS"]
    tbl = tbl.sort_values(['valset', 'method', 'dataset'])
    print(tbl.info())
    print('=================================================================')
    print(tbl.to_csv(index=False))


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    files_dir = args.dir
    std_perc_report = args.std_perc
    img_hist_report = args.img_hist
    heatmap = args.hm
    mnemonic = args.model
    scaler = args.sc
    dataset = args.dataset
    metric = args.metric
    gen_table = args.gen_table
    gen_trace_metrics_table = args.gen_trace_table
    dzm_bias = args.dzm_bias
    residual_plot = args.residual_plot
    valset = args.valset
    epochs = args.e
    run = args.run
    pred_file = args.pf
    save = args.save

    if std_perc_report:
        gen_std_perc_report(files_dir)

    if img_hist_report:
        gen_img_hist_report(files_dir, mnemonic, scaler, dataset, metric, epochs, run, save=save)

    if heatmap:
        gen_heatmap_report(pred_file, files_dir, mnemonic, scaler, dataset, valset)

    if gen_table:
        gen_table_report(files_dir)

    if gen_trace_metrics_table:
        gen_trace_table_report(files_dir)

    if dzm_bias:
        dzm_bias_report(files_dir)

    if residual_plot:
        residual_plot_report(pred_file, files_dir, mnemonic, scaler, dataset, valset)



