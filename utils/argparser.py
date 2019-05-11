import argparse

def get_args_training():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--name_experiment", type=str, required=True,
        help="Model to train.",
    )
    parser.add_argument(
        "-e", "--nbrepoch", type=int, default=30,
        help="Number of epoch.",
    )
    parser.add_argument(
        "-b", "--batchsize", type=int, default=32,
        help="Size of the batch.",
    )
    parser.add_argument(
        "-v", "--validation", type=str, default="datas/Val",
        help="Directory containing data to validate.",
    )
    parser.add_argument(
        "-t", "--training", type=str, default="datas/Train",
        help="Directory containing data to train.",
    )
    # parser.add_argument(
    #     "-w", "--weights", type=str, default=None,
    #     help="weights.",
    # )
    args = parser.parse_args()
    return args.name_experiment, args.nbrepoch, args.batchsize, args.validation, args.training
