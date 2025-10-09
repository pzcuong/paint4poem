import argparse
import csv
import os
import pickle
import re
from typing import Optional

import numpy as np
from nltk.tokenize import RegexpTokenizer


def _extract_data_dir(cfg_path: str) -> str:
    """Return the DATA_DIR value from a simple YAML config."""
    with open(cfg_path, encoding='utf-8') as cfg_file:
        for raw_line in cfg_file:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('DATA_DIR'):
                _, value = line.split(':', 1)
                value = value.strip().strip("'\"")
                if not value:
                    break
                return value
    raise ValueError(f"Could not find DATA_DIR in config: {cfg_path}")


def _resolve_data_dir(data_dir: Optional[str], cfg: Optional[str]) -> str:
    if cfg:
        resolved = _extract_data_dir(cfg)
        if data_dir and data_dir != resolved:
            print(f"Overriding DATA_DIR from config ({resolved}) with CLI value: {data_dir}")
            return os.path.expanduser(data_dir)
        return os.path.expanduser(resolved)
    if not data_dir:
        raise ValueError('A dataset directory must be provided via --data_dir or --cfg.')
    return os.path.expanduser(data_dir)


def _maybe_find_metadata_csv(data_dir: str) -> Optional[str]:
    """Return a metadata CSV filename if one exists in the dataset directory."""
    candidates = ['TITLE-IMAGE.csv', 'POEM-IMAGE.csv']
    for candidate in candidates:
        candidate_path = os.path.join(data_dir, candidate)
        if os.path.isfile(candidate_path):
            return candidate

    try:
        for entry in os.listdir(data_dir):
            upper = entry.upper()
            if upper.startswith('TITLE-IMAGE') and entry.lower().endswith('.csv'):
                return entry
            if upper.startswith('POEM-IMAGE') and entry.lower().endswith('.csv'):
                return entry
    except FileNotFoundError:
        pass
    return None


def title_image_prep(data_dir, csv_name):
    text_path = os.path.join(data_dir, 'text')
    if not os.path.isdir(text_path):
        os.makedirs(text_path)

    max_lth = 0
    lth_ls = []  # length of lists
    with open(os.path.join(data_dir, csv_name), encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        files = []
        cnt_empty = 0
        cnt_english = 0
        for index, row in enumerate(csv_reader):
            file = row[12]
            cap = row[5]
            # check if string contains chinese characters
            if re.search(u'[\u4e00-\u9fff]', cap):
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                tokens = [j for i in tokens for j in i]
                lth_ls.append(len(tokens))
                if len(tokens) > max_lth:
                    max_lth = len(tokens)
                    print(tokens)
                # check if tokens contain more than one english character
                if sum([t.isascii() for t in tokens]) > 1:
                    cnt_english += 1
                    print(index, tokens)
                else:
                    files.append(file)
                    with open(text_path + '/' + file + '.txt', 'w', encoding='utf-8') as txt:
                        txt.write(cap)
            elif cap == '':
                cnt_empty += 1
            else:
                cnt_english += 1
    print(lth_ls)

    print('total amount english (delete this from data): {}'.format(cnt_english))
    print('total amount empty (delete this from data): {}'.format(cnt_empty))
    print(max_lth)
    return files


def main(args):
    data_dir = os.path.abspath(args.data_dir)
    print(f"Using dataset directory: {data_dir}")

    train_path = os.path.join(data_dir, 'train')
    if not os.path.isdir(train_path):
        os.makedirs(train_path)

    test_path = os.path.join(data_dir, 'test')
    if not os.path.isdir(test_path):
        os.makedirs(test_path)

    csv_name = _maybe_find_metadata_csv(data_dir)
    if csv_name:
        print(f"Found metadata CSV '{csv_name}'. Generating text splits from CSV contents.")
        filenames = title_image_prep(data_dir, csv_name)
    else:
        filenames = os.listdir(os.path.join(data_dir, 'text'))
        filenames = [name.rpartition('.txt')[0] for name in filenames]

    print('total amount of data: {}'.format(len(filenames)))
    np.random.seed(seed=1)
    np.random.shuffle(filenames)
    split_idx = int(0.75 * len(filenames))
    train_files, test_files = filenames[:split_idx], filenames[split_idx:]
    print('total amount of training data: {}'.format(len(train_files)))
    print('total amount of test data: {}'.format(len(test_files)))

    with open(os.path.join(train_path, 'filenames.pickle'), 'wb') as f:
        pickle.dump(train_files, f, protocol=2)

    with open(os.path.join(test_path, 'filenames.pickle'), 'wb') as f:
        pickle.dump(test_files, f, protocol=2)

    with open(os.path.join(test_path, 'filenames.pickle'), 'rb') as f:
        test = pickle.load(f)
        print('10 test filenames:')
        print(test[:10])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', type=str, default=None,
                        help='Path to the dataset directory containing text/ and image folders.')
    parser.add_argument('--cfg', dest='cfg', type=str, default=None,
                        help='Optional YAML config file that specifies DATA_DIR.')
    args = parser.parse_args()
    args.data_dir = _resolve_data_dir(args.data_dir, args.cfg)
    main(args)
