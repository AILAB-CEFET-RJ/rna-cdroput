import argparse
import glob


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments')
    parse.add_argument('-n', metavar='NAME', help='Experiment name. This name is included in output files.')

    return parse


def select_best_model(name):
    file_mask = 'output/models/epochs_*/run_*/model_*'
    if name:
        file_mask = f"{file_mask[:-1]}{name}_*"
    model_files = glob.glob(file_mask)
    models = {}

    for mf in model_files:
        key = float(mf.split('mse_')[-1].split('_')[0])
        models[key] = mf

    models = {k: models[k] for k in sorted(models)}
    return list(models.values())[0]


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    best_model = select_best_model(args.n)

    print(best_model)
