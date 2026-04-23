import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves from a log file.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the log file.")
    parser.add_argument("--postfix", type=str, default="", help="Custom string to append to the output filename.")
    return parser.parse_args()


def parse_log(log_path: str):
    epochs = []
    train_losses = []
    train_qwks = []
    val_losses = []
    val_qwks = []

    with open(log_path, "r") as f:
        for line in f:
            parts = [col.strip() for col in line.strip().split("|")]
            if len(parts) < 8:
                continue
            mode = parts[1]
            if mode == "Train":
                epoch_str = parts[0].strip()
                epoch = int(epoch_str.split("/")[0])
                epochs.append(epoch)
                train_losses.append(float(parts[2]))
                train_qwks.append(float(parts[7]))
            elif mode == "Val":
                val_losses.append(float(parts[2]))
                val_qwks.append(float(parts[7]))

    return epochs, train_losses, train_qwks, val_losses, val_qwks


def plot_curves(epochs, train_losses, train_qwks, val_losses, val_qwks, output_path: Path):
    fig, (ax_loss, ax_qwk) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left subplot: Loss ---
    ax_loss.plot(epochs, train_losses, color="tab:red", marker="o", markersize=4, label="Train")
    val_epochs = epochs[: len(val_losses)]
    ax_loss.plot(val_epochs, val_losses, color="tab:blue", marker="s", markersize=4, label="Validation")
    ax_loss.set_title("(a) Loss across epochs")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.6)

    # --- Right subplot: QWK ---
    ax_qwk.plot(epochs, train_qwks, color="tab:red", marker="o", markersize=4, label="Train")
    val_epochs = epochs[: len(val_qwks)]
    ax_qwk.plot(val_epochs, val_qwks, color="tab:blue", marker="s", markersize=4, label="Validation")
    ax_qwk.set_title("(b) QWK across epochs")
    ax_qwk.set_xlabel("Epochs")
    ax_qwk.set_ylabel("Quadratic Weighted Kappa (QWK)")
    ax_qwk.legend()
    ax_qwk.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main():
    args = parse_args()
    epochs, train_losses, train_qwks, val_losses, val_qwks = parse_log(args.log_path)

    if args.postfix:
        filename = f"figure_5_1_{args.postfix}.png"
    else:
        filename = "figure_5_1.png"

    root = Path(__file__).resolve().parents[2]
    output_path = root / "results" / filename

    plot_curves(epochs, train_losses, train_qwks, val_losses, val_qwks, output_path)


if __name__ == "__main__":
    main()

