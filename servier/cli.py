import argparse

from servier.config import config
from servier.model.evaluate import evaluate
from servier.model.predict import predict
from servier.model.train import train
from servier.schemas import EvaluateArgs, PredictArgs, TrainArgs


def train_cli(args):
    train(config=config, args=TrainArgs(**vars(args)))


def evaluate_cli(args):
    evaluate(config=config, args=EvaluateArgs(**vars(args)))


def predict_cli(args):
    predict(config=config, args=PredictArgs(**vars(args)))


def cli():
    parser = argparse.ArgumentParser(description="Servier command-line tool")

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        dest="data_path",
        help="Path to training data",
    )

    train_parser.set_defaults(func=train_cli)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        dest="data_path",
        help="Path to training data",
    )
    evaluate_parser.set_defaults(func=evaluate_cli)

    predict_parser = subparsers.add_parser(
        "predict", help="Make predictions with the model"
    )
    predict_parser.add_argument(
        "--input", type=str, required=True, help="Input data for prediction"
    )
    predict_parser.set_defaults(func=predict_cli)

    args = parser.parse_args()

    if args.subcommand is not None:
        return args.func(args)

    return parser.print_help()
