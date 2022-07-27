import argparse
import glob


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments')
    parse.add_argument('-n', metavar='NAME', help='Experiment name. This name is included in output files.')
    parse.add_argument('-e', metavar='EPOCHS', type=int, help='Epochs.')
    parse.add_argument('-s', action='store_true', help='Return a single best model between across the runs')

    return parse


def select_best_model(name, epochs):
    file_mask = f"output/models/epochs_{epochs}/run_*/model_*"
    if name:
        file_mask = f"{file_mask[:-1]}{name}_*"
    model_files = glob.glob(file_mask)
    models = {}

    for mf in model_files:
        key = float(mf.split('mse_')[-1].split('_')[0])
        models[key] = mf

    models = {k: models[k] for k in sorted(models)}
    return list(models.values())[0]


def select_best_models(name, epochs):
    run_dir_file_mask = f"output/models/epochs_{epochs}/run_*"
    run_dirs = glob.glob(run_dir_file_mask)
    bests = []

    for run in run_dirs:
        file_mask = f"{run}/model_*"
        if name:
            file_mask = f"{file_mask[:-1]}{name}_*"

        model_files = glob.glob(file_mask)
        models = {}

        for mf in model_files:
            key = float(mf.split('mse_')[-1].split('_')[0])
            models[key] = mf

        models = {k: models[k] for k in sorted(models)}
        bests.append(list(models.values())[0])

    return " ".join(bests)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    if not args.e:
        raise Exception('ERROR: -e option cant be empty!')

    if args.s:
        best_model = select_best_model(args.n, args.e)
    else:
        best_model = select_best_models(args.n, args.e)

    print(best_model)
