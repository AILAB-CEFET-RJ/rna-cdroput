import argparse
import math
import pandas as pd


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments')
    parse.add_argument('-e', metavar='EPOCHS', type=int, help='Epochs.')
    parse.add_argument('-dp', metavar='DROPOUT', help='Dropout class to use.')
    parse.add_argument('-runs', metavar='RUNS', type=int, help='Total runs.')
    parse.add_argument('-lr', metavar='LR', type=float, help='Learning rate.')
    parse.add_argument('-trainset', metavar='TRAINSET', help='Train dataset file.')
    parse.add_argument('-testset', metavar='TESTSET', help='Test dataset file.')
    parse.add_argument('-valset', metavar='VALSET', help='Validation dataset file.')
    parse.add_argument('-gpu', metavar='DEVICE', help='GPU device name. Default is device name position 0.')
    parse.add_argument('-bs', metavar='BATCH', type=int, default=0, help='Batch size.')
    parse.add_argument('-layers', metavar='LAYERS', help='Force amount of units in each hidden layer. '
       'Use "20:10" value for 2 hidden layers with 20 neurons in first and 10 in seccond. The Default is 2 hidden layers'
       ' and neurons are computed based on the size of the features.')

    return parse


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    features = 'u,g,r,i,z'.split(',')
    f = len(features)

    train_df = pd.read_csv(args.trainset, comment='#')
    print(f"Train set loaded! Shape = {train_df.shape}")

    if args.layers is None:
        m = 1
        N = train_df.shape[0]
        l1 = round(math.sqrt((m + 2) * N) + 2 * math.sqrt(N / (m + 2)))
        l2 = round(m * math.sqrt(N / (m + 2)))
        args.layers = f"{l1}:{l2}"
        print(f"Using layers= {args.layers}")


