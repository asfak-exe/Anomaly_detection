import os
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================================================
# CONFIG
# ==========================================================

RUN_DIR = r"C:\Users\6000_J1I\Desktop\asfak_project\PADS_AE\FINAL\output\run_10"

PREDICTIONS_CSV = os.path.join(
    RUN_DIR,
    "predictions.csv"
)

SENSOR_THRESHOLD_CSV = os.path.join(
    RUN_DIR,
    "sensor_thresholds.csv"
)

OUTPUT_DIR = os.path.join(
    RUN_DIR,
    "plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

TIMESTAMP_COL = "Timestamp"
ANOMALY_COL = "is_anomaly"
LOSS_COL = "Reconstruction_Loss"

WINDOW_HOURS = 1

def load_data():
    """
    Loads predictions and sensor thresholds.
    """

    predictions_df = pd.read_csv(PREDICTIONS_CSV)

    predictions_df[TIMESTAMP_COL] = pd.to_datetime(
        predictions_df[TIMESTAMP_COL]
    )

    threshold_df = pd.read_csv(
        SENSOR_THRESHOLD_CSV
    )

    sensor_thresholds = dict(
        zip(
            threshold_df["Sensor"],
            threshold_df["Threshold"]
        )
    )

    return predictions_df, sensor_thresholds

def get_top_sensor(row):
    """
    Returns the sensor having the largest
    scaled reconstruction error.
    """

    error_cols = [
        c for c in row.index
        if c.endswith("_Error")
    ]

    top_error_col = row[error_cols].idxmax()

    top_sensor = top_error_col.replace(
        "_Error",
        ""
    )

    top_error = row[top_error_col]

    return top_sensor, top_error


def find_event_bounds(
    predictions_df,
    timestamp,
    sensor,
    threshold,
    tolerance=2,
):
    """
    Finds the complete threshold crossing event.

    Parameters
    ----------
    tolerance : int
        Number of consecutive samples below threshold
        required to terminate the event.
    """

    anomaly_idx = predictions_df.index[
        predictions_df[TIMESTAMP_COL] == timestamp
    ][0]

    # --------------------------------------------------
    # Search LEFT
    # --------------------------------------------------

    start_idx = anomaly_idx
    below_count = 0

    while start_idx > 0:

        if predictions_df.loc[start_idx - 1, sensor] >= threshold:
            below_count = 0
            start_idx -= 1
        else:
            below_count += 1

            if below_count >= tolerance:
                break

            start_idx -= 1

    # --------------------------------------------------
    # Search RIGHT
    # --------------------------------------------------

    end_idx = anomaly_idx
    below_count = 0

    while end_idx < len(predictions_df) - 1:

        if predictions_df.loc[end_idx + 1, sensor] >= threshold:
            below_count = 0
            end_idx += 1
        else:
            below_count += 1

            if below_count >= tolerance:
                break

            end_idx += 1

    # --------------------------------------------------
    # Peak
    # --------------------------------------------------

    peak_idx = predictions_df.loc[
        start_idx:end_idx,
        sensor
    ].idxmax()

    return (
        start_idx,
        peak_idx,
        end_idx,
    )


def plot_anomaly(
    predictions_df,
    timestamp,
    sensor,
    threshold,
    event_id,
):
    """
    Plot one anomaly using the top contributing sensor.
    """

    # --------------------------------------------------
    # Create 2-hour window
    # --------------------------------------------------

    start_time = timestamp - pd.Timedelta(hours=WINDOW_HOURS)
    end_time = timestamp + pd.Timedelta(hours=WINDOW_HOURS)

    plot_df = predictions_df[
        (predictions_df[TIMESTAMP_COL] >= start_time) &
        (predictions_df[TIMESTAMP_COL] <= end_time)
    ].copy()

    if plot_df.empty:
        return

    # --------------------------------------------------
    # Sensor Values
    # --------------------------------------------------

    values = plot_df[sensor]

    # --------------------------------------------------
# Auto-scale Y-axis
# --------------------------------------------------

    ymin = values.min()
    ymax = values.max()

    if ymax == ymin:
      padding = max(abs(ymax) * 0.05, 1)
    else:
      padding = 0.05 * (ymax - ymin)

    # --------------------------------------------------
    # Create Figure
    # --------------------------------------------------

    plt.figure(figsize=(16,6))

    plt.plot(
        plot_df[TIMESTAMP_COL],
        values,
        color="royalblue",
        linewidth=2,
        label=sensor,
    )

    # --------------------------------------------------
    # Threshold
    # --------------------------------------------------

    plt.axhline(
        threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold:.2f})",
    )

    # --------------------------------------------------
    # Highlight Above Threshold
    # --------------------------------------------------

    mask = values.ge(threshold)

    plt.fill_between(
        plot_df[TIMESTAMP_COL],
        values,
        threshold,
        where=mask,
        interpolate=True,
        color="red",
        alpha=0.35,
        label="Above Threshold",
    )

    plt.ylim(
    ymin - padding,
    ymax + padding,
)

    # --------------------------------------------------
    # Mark anomaly timestamp
    # --------------------------------------------------

    anomaly_value = predictions_df.loc[
        predictions_df[TIMESTAMP_COL] == timestamp,
        sensor,
    ].iloc[0]

    plt.scatter(
        timestamp,
        anomaly_value,
        color="black",
        marker="*",
        s=180,
        zorder=5,
        label="Detected Anomaly",
    )

    plt.axvline(
        timestamp,
        color="black",
        linestyle=":",
        alpha=0.6,
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    plt.title(
        f"Event {event_id:03d} | {sensor}",
        fontsize=15,
        fontweight="bold",
    )

    plt.xlabel("Timestamp")
    plt.ylabel(sensor)

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )

    plt.xticks(rotation=30)

    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    filename = f"Event_{event_id:03d}_{sensor}.png"

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def main():

    # ------------------------------------------
    # Load data
    # ------------------------------------------

    predictions_df, sensor_thresholds = load_data()

    anomaly_df = predictions_df[
        predictions_df[ANOMALY_COL] == 1
    ].copy()

    print(f"Detected {len(anomaly_df)} anomalies.")

    # ------------------------------------------
    # Generate plots
    # ------------------------------------------

    for event_id, (_, row) in enumerate(anomaly_df.iterrows(), start=1):

        timestamp = row[TIMESTAMP_COL]

        top_sensor, top_error = get_top_sensor(row)

        if top_sensor not in sensor_thresholds:
            print(f"Skipping {top_sensor} (threshold not found)")
            continue

        threshold = sensor_thresholds[top_sensor]

        print(
            f"[{event_id}/{len(anomaly_df)}] "
            f"{timestamp} | "
            f"{top_sensor} | "
            f"Error={top_error:.3f}"
        )

        plot_anomaly(
            predictions_df=predictions_df,
            timestamp=timestamp,
            sensor=top_sensor,
            threshold=threshold,
            event_id=event_id,
        )

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()


