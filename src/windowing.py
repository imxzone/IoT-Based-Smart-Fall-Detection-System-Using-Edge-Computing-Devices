import pandas as pd
from src.config import WINDOW_SIZE, STEP_SIZE


def create_windows(df, subject, activity, trial, label, split, source):
    windows = []
    window_id = 0

    for start in range(0, len(df), STEP_SIZE):
        end = start + WINDOW_SIZE
        window_df = df.iloc[start:end].copy()

        if len(window_df) < WINDOW_SIZE // 2:
            break

        if len(window_df) < WINDOW_SIZE:
            padding_size = WINDOW_SIZE - len(window_df)
            zero_padding = pd.DataFrame(0, index=range(padding_size), columns=df.columns)
            window_df = pd.concat([window_df, zero_padding], ignore_index=True)
        else:
            window_df = window_df.reset_index(drop=True)

        windows.append({
            "window_id": window_id,
            "source": source,
            "subject": subject,
            "activity": activity,
            "trial": trial,
            "label": label,
            "split": split,
            "start_index": start,
            "end_index": min(end, len(df)) - 1,
            "data": window_df,
        })

        window_id += 1

    return windows