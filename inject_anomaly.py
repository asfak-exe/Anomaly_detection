import os
import json
import random
import numpy as np
import pandas as pd



# CONFIG


DATA_PATH = r"C:\Users\6000_J1I\Desktop\asfak_project\data_from_feb_2026.csv"

OUTPUT_DIR = r"C:\Users\6000_J1I\Desktop\asfak_project\PADS_AE\synthetic_tests"

ANOMALY_TYPE = "spike"
# spike
# drift
# stuck
# noise
# mixed

START_TIME = "2026-04-15 08:00:00"
END_TIME   = "2026-04-15 15:00:00"

NUM_TAGS = 5

MAGNITUDE = 20

RANDOM_SEED = 42



def load_data(data_path):

    df = pd.read_csv(data_path)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    feature_columns = [
        c for c in df.columns
        if c != "Timestamp"
    ]

    print("="*60)
    print("Dataset Information")
    print("="*60)

    print(f"Rows        : {len(df)}")
    print(f"Features    : {len(feature_columns)}")
    print(f"Start Time  : {df['Timestamp'].min()}")
    print(f"End Time    : {df['Timestamp'].max()}")

    return df, feature_columns


def select_window(df):

    mask = (
        (df["Timestamp"] >= START_TIME)
        &
        (df["Timestamp"] <= END_TIME)
    )

    window_idx = df.index[mask]

    if len(window_idx) == 0:
        raise ValueError(
            "No rows found in selected time window."
        )

    print(f"\nRows inside window : {len(window_idx)}")

    return window_idx


def choose_tags(feature_columns):

    random.seed(RANDOM_SEED)

    selected = random.sample(
        feature_columns,
        NUM_TAGS
    )

    print("\nAffected Tags")

    for tag in selected:
        print(tag)

    return selected

def inject_spike(
    df,
    window_idx,
    selected_tags,
):

    for tag in selected_tags:

        std = df[tag].std()

        direction = random.choice([-1, 1])

        df.loc[window_idx, tag] += (
            direction * MAGNITUDE * std
        )

    return df

def inject_drift(
    df,
    window_idx,
    selected_tags,
):

    n = len(window_idx)

    for tag in selected_tags:

        std = df[tag].std()

        drift = np.linspace(
            0,
            MAGNITUDE * std,
            n
        )

        direction = random.choice([-1, 1])

        df.loc[window_idx, tag] += (
            direction * drift
        )

    return df


def inject_stuck(
    df,
    window_idx,
    selected_tags,
):

    for tag in selected_tags:

        value = df.loc[
            window_idx[0],
            tag
        ]

        df.loc[
            window_idx,
            tag
        ] = value

    return df


def inject_noise(
    df,
    window_idx,
    selected_tags,
):

    for tag in selected_tags:

        std = df[tag].std()

        noise = np.random.normal(
            0,
            MAGNITUDE * std,
            len(window_idx)
        )

        df.loc[
            window_idx,
            tag
        ] += noise

    return df


def inject_mixed(
    df,
    window_idx,
    selected_tags,
):

    funcs = [
        inject_spike,
        inject_drift,
        inject_noise,
        inject_stuck,
    ]

    for tag in selected_tags:

        random.choice(funcs)(
            df,
            window_idx,
            [tag],
        )

    return df

def save_files(
    df,
    window_idx,
    output_dir,
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    
    # Synthetic Test Data
    


    synthetic_path = os.path.join(
        output_dir,
        "synthetic_test.csv"
    )

    df.to_csv(
        synthetic_path,
        index=False
    )

   
    # Ground Truth
    

    ground_truth = pd.DataFrame({

        "Timestamp": df["Timestamp"],

        "is_anomaly": 0,

        "anomaly_type": "normal"

    })

    ground_truth.loc[
        window_idx,
        "is_anomaly"
    ] = 1

    ground_truth.loc[
        window_idx,
        "anomaly_type"
    ] = ANOMALY_TYPE

    ground_truth_path = os.path.join(
        output_dir,
        "ground_truth.csv"
    )

    ground_truth.to_csv(
        ground_truth_path,
        index=False
    )

    return (
        synthetic_path,
        ground_truth_path
    )


def save_report(
    output_dir,
    selected_tags,
    window_idx,
    df,
):

    report = {

        "anomaly_type": ANOMALY_TYPE,

        "start_time": str(
            df.loc[
                window_idx[0],
                "Timestamp"
            ]
        ),

        "end_time": str(
            df.loc[
                window_idx[-1],
                "Timestamp"
            ]
        ),

        "rows_modified": len(window_idx),

        "affected_tags": selected_tags,

        "magnitude": MAGNITUDE

    }

    report_path = os.path.join(
        output_dir,
        "injection_report.json"
    )

    with open(
        report_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    return report_path


if __name__ == "__main__":

    df, feature_columns = load_data(
        DATA_PATH
    )

    window_idx = select_window(
        df
    )

    selected_tags = choose_tags(
        feature_columns
    )

    if ANOMALY_TYPE == "spike":

        df = inject_spike(
            df,
            window_idx,
            selected_tags
        )

    elif ANOMALY_TYPE == "drift":

        df = inject_drift(
            df,
            window_idx,
            selected_tags
        )

    elif ANOMALY_TYPE == "noise":

        df = inject_noise(
            df,
            window_idx,
            selected_tags
        )

    elif ANOMALY_TYPE == "stuck":

        df = inject_stuck(
            df,
            window_idx,
            selected_tags
        )

    elif ANOMALY_TYPE == "mixed":

        df = inject_mixed(
            df,
            window_idx,
            selected_tags
        )

    else:

        raise ValueError(
            "Unknown anomaly type."
        )

    synthetic_path, ground_truth_path = save_files(
        df,
        window_idx,
        OUTPUT_DIR
    )

    report_path = save_report(
        OUTPUT_DIR,
        selected_tags,
        window_idx,
        df
    )

    print("\n" + "=" * 60)
    print("Injection Complete")
    print("=" * 60)

    print(f"Type           : {ANOMALY_TYPE}")
    print(f"Rows Modified  : {len(window_idx)}")
    print(f"Tags Modified  : {len(selected_tags)}")

    print("\nAffected Tags")

    for tag in selected_tags:

        print(f" - {tag}")

    print(f"\nSynthetic Data : {synthetic_path}")
    print(f"Ground Truth   : {ground_truth_path}")
    print(f"Report         : {report_path}")