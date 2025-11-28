import os 
import pandas as pd
import matplotlib.pyplot as plt 


RESULTS_CSV = os.path.join("results", "metrics_history.csv")
PLOTS_DIR = os.path.join("results", "plots_history")

def save_plot(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot: {filename}")


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"No results file found at {RESULTS_CSV}. Run train_models.py first.")
        return



    df = pd.read_csv(RESULTS_CSV)

    print("loaded metrics history:")
    print(df.head())

    #plot roc auc over runs for each model if available
    if "roc_auc" in df.columns:
        fig = plt.figure()
        for model_name, group in df.groupby("model"):
            group = group.reset_index(drop=True)
            plt.plot(group.index, group['roc_auc'], marker='o', label=model_name)

        plt.xlabel("run number (per model)")
        plt.ylabel("roc-auc")
        plt.title("roc auc over time for each model")
        plt.legend()
        plt.tight_layout()
        save_plot(fig, "roc_auc_over_time.png")
        plt.show()

    if "accuracy" in df.columns:
        fig = plt.figure()
        for model_name, group in df.groupby("model"):
            group = group.reset_index(drop=True)
            plt.plot(group.index, group["accuracy"], marker="o", label=model_name)

        plt.xlabel("Run number (per model)")
        plt.ylabel("Accuracy")
        plt.title("accuracy over time for each model")
        plt.legend()
        plt.tight_layout()
        save_plot(fig, "accuracy_over_time.png")
        plt.show()


if __name__ == "__main__":
    main()