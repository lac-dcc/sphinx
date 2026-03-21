import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau, spearmanr
import logging
import sys

from config.params import params

# --- CONFIGURATION ---
METRICS_PATH = params.paths.labels
TRAIN_SPLIT = params.paths.train_graphs_txt
TEST_SPLIT = params.paths.test_graphs_txt


def log_blank_line():
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'stream'):
            handler.stream.write('\n')
            handler.flush()


def load_data(split_file, metrics_data):
    x = []
    y = []
    skipped = 0

    with open(split_file, 'r') as f:
        filenames = [line.strip().replace('.pt', '') for line in f if line.strip()]

    for name in filenames:
        if name in metrics_data:
            if 'trainable_parameters' in metrics_data[name]:
                trainable_params = metrics_data[name]['trainable_parameters']
                metric = metrics_data[name][params.model.target_performance_metric]
                x.append(trainable_params)
                y.append(metric)
            else:
                skipped += 1
        else:
            skipped += 1

    if skipped > 0:
        logging.info(f"Warning: Skipped {skipped} entries (missing data in JSON).")

    return np.array(x).reshape(-1, 1), np.array(y)


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"Loading metrics from {METRICS_PATH}...")
    with open(METRICS_PATH, 'r') as f:
        metrics_data = json.load(f)

    log_blank_line()
    logging.info("Loading Training Data...")
    x_train, y_train = load_data(TRAIN_SPLIT, metrics_data)
    logging.info(f"Train Samples: {len(x_train)}")

    log_blank_line()
    logging.info("Loading Test Data...")
    x_test, y_test = load_data(TEST_SPLIT, metrics_data)
    logging.info(f"Val Samples:   {len(x_test)}")

    if len(x_train) == 0:
        logging.info("Error: Training data empty. Check JSON keys or file paths.")
        return

    log_blank_line()
    logging.info(f"Training Linear Regression ({params.model.target_performance_metric} ~ Parameters)...")
    model = LinearRegression()
    model.fit(x_train, y_train)

    logging.info(f"Equation: {params.model.target_performance_metric} = {model.coef_[0]:.2e} * Params + {model.intercept_:.4f}")

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    tau, _ = kendalltau(y_test, y_pred)
    spearman, _ = spearmanr(y_test, y_pred)

    log_blank_line()
    logging.info("LINEAR BASELINE RESULTS")
    logging.info(f"RMSE:           {rmse:.5f}")
    logging.info(f"MAE:            {mae:.5f}")
    logging.info(f"R²:             {r2:.5f}")
    logging.info(f"Kendall's Tau:  {tau:.5f}")
    logging.info(f"Spearman's Rho: {spearman:.5f}")


if __name__ == "__main__":
    main()