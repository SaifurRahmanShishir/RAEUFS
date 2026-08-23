
import argparse
from pathlib import Path
import scipy
import sys
sys.path.insert(0, 'data')
sys.path.insert(0, 'src')
import numpy as np
import pytorch_lightning as pl

from data import CustomDataset, load_jaffe, scale_features
from evaluation import EVMetrics, baseline_performance
from model import RSRAutoEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train RAEUFS")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--selected-features", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda1", type=float, default=0.001)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--d-bar", type=int, default=50)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "auto"],
        default="cpu",
        help="CPU is the safest default on Apple Silicon for this implementation.",
    )
    return parser.parse_args()


def main():


    # Load the .mat file
    data = scipy.io.loadmat('jaffe.mat')

    # Extract features and labels
    X = data['fea']            # Features
    y = data['gnd'].flatten()  # Labels

    # Convert into a PyTorch Dataset
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    print(f"X shape: {X.shape}; clusters: {len(np.unique(y))}")

    X_scaled = scale_features(X, method="minmax")
    train_dataset = CustomDataset(X_scaled)
    print("Baseline [ACC mean %, ACC std %, NMI mean %, NMI std %]:")
    print(baseline_performance(X_scaled, y, X_scaled, y))

    acc_tr = np.zeros(args.runs)
    nmi_tr = np.zeros(args.runs)

    for run in range(args.runs):
        pl.seed_everything(args.seed + run, workers=True)
        print(f"\n=== Run {run + 1}/{args.runs} ===")

        model = RSRAutoEncoder(
            input_dim=X.shape[1],
            selected_features=args.selected_features,
            d_A=len(np.unique(y)),
            d_bar=args.d_bar,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            theta=args.theta,
            sigma=args.sigma,
            train_data=train_dataset,
            val_data=None,
            update_S_every=1,
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=args.accelerator,
            devices=1,
            enable_progress_bar=True,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model)

        metric = EVMetrics(
            model,
            X_scaled.detach().cpu().numpy(),
            y,
            args.selected_features,
        )
        acc_tr[run], nmi_tr[run] = metric.kmeans_cluster_accuracy_nmi()
        print(f"ACC={acc_tr[run]:.6f}, NMI={nmi_tr[run]:.6f}")

    print("\n=== Summary ===")
    print(f"ACC mean: {np.mean(acc_tr):.6f}")
    print(f"ACC std : {np.std(acc_tr):.6f}")
    print(f"NMI mean: {np.mean(nmi_tr):.6f}")
    print(f"NMI std : {np.std(nmi_tr):.6f}")


if __name__ == "__main__":
    main()
