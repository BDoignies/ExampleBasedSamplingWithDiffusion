from utils.Config import ParseTrainConfig, parse_eval
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-c", "--config", help="Path to config file", required=True, type=str)
    parser.add_argument("--its", help="Number of optimization steps", required=False, type=int, default=int(1e10))
    parser.add_argument("-t", "--time", help="Training time in minutes", required=False, default=180, type=bool)
    parser.add_argument("--tqdm", help="Adds progress bar for training", required=False, default=True, type=bool)

    args = parser.parse_args()

    trainer = ParseTrainConfig(args.config)

    # Lowest of both 'args.its, args.time' will be used
    trainer.train(args.its, args.tqdm, args.time)