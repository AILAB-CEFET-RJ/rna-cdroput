import argparse
import pandas as pd

from numpy.random import seed
import matplotlib.pyplot as plt

from modules import dataset_handle as dh


def parser():
   parser = argparse.ArgumentParser(description='Reports module')
   parser.add_argument('-dataset', metavar='DATASET', help='Dataset file.')
   parser.add_argument('-ref_file', metavar='REF', help='Result ref id file.')
   parser.add_argument('-output', metavar='OUT', help='Result file.')

   return parser


if __name__ == '__main__':
    seed(42)
    parser = parser()
    args = parser.parse_args()

    df = dh.load_test_dataframe(args.dataset)
    df_ref_id = pd.read_csv(args.ref_file)

    ref_id_list = []
    with open(args.output, 'r') as file:
        preds_found = False
        for line in file:
            if '##PREDS INIT' in line:
                preds_found = True
                continue

            if '##PREDS END' in line:
                preds_found = False

            if '[Prediction Outliers]' in line:
                continue

            if preds_found:
                v = line.replace('id: [', '') \
                    .replace('[', '') \
                    .replace('\n', '') \
                    .replace(']', '').strip().split(' ')
                v = [int(i) for i in v]
                ref_id_list.append(v)

    outliers = {}
    i=5
    for b in 'ugriz':
        outliers[b] = []
        for ref_id_arr in ref_id_list:
            orefid = ref_id_arr[i]
            if orefid:
                id = df_ref_id[df_ref_id['refid'] == orefid]['objid'].values[0]
                outliers[b].append(id)
        i+=1

    df_outl_pband = {}
    for b in 'ugriz':
        df_outl = df[['refid', b, f"err_{b}"]]
        df_outl = pd.merge(df_outl, df_ref_id, on='refid')[['objid', b, f"err_{b}"]]
        df_outl.insert(3, 'outlier', df_outl[['objid']].isin(outliers[b]))
        df_outl_pband[b] = df_outl
        print(f"Band [{b}]:")
        total = df_outl.shape[0]
        oulrs = df_outl[df_outl['outlier'] == True].shape[0]
        print(f"\t total = [{total}]:")
        print(f"\t outliers = [{oulrs}]:")
        print(f"\t outliers rate = {(oulrs/total) * 100:.2f}%")
        print('--------------------------')
        df_outl.to_csv(f"outliers_{b}_band.csv")

    print('Ploting ...')
    for b in 'ugriz':
        df_aux = df_outl_pband[b][[b, f"err_{b}", 'outlier']]
        ax = df_aux[df_aux['outlier'] == False].plot.scatter(x=b, y=f"err_{b}", c='b')
        df_aux[df_aux['outlier'] == True].plot.scatter(x=b, y=f"err_{b}", c='r', ax=ax)
        plt.show()
        plt.savefig(f"outliers_{b}_band.png")
    print('Done')
