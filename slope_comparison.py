import argparse, json
import numpy as np
import pandas as pd
import tqdm


def load_slopes(csv_path, label):
    df = pd.read_csv(csv_path)
    # parse JSON‑serialized slope arrays
    df[label] = df['slopes'].apply(json.loads).apply(np.array)
    df['key']   = df['speaker'] + '/' + df['video'].fillna('') + '/' + df['file']
    return df.set_index('key')[[label]]

def main(csvs, labels, out_csv):
    # load each CSV’s slope column under its label, then align on key
    dfs    = [load_slopes(p, l) for p, l in zip(csvs, labels)]
    merged = dfs[0].join(dfs[1:], how='inner')
    base   = labels[0]
    rows   = []
    for key, row in tqdm.tqdm(merged.iterrows()):
        b = row[base]
        for other in labels[1:]:
            c = row[other]
            L = min(len(b), len(c))
            mae = np.mean(np.abs(c[:L] - b[:L])) if L > 0 else np.nan
            rows.append({'key': key, 'compare_level': other, 'mae_slope': mae})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csvs',   nargs='+', required=True, help='feature CSVs (baseline first)') # link to all the csv files
    p.add_argument('--labels', nargs='+', required=True, help='labels for each CSV') # original for the baseline csv (the unanonymized dataset)
    p.add_argument('--out_csv', default='slope_mae.csv', help='output CSV path')
    args = p.parse_args()
    main(args.csvs, args.labels, args.out_csv)
