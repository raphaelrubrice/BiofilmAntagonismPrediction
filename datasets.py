import time
import pandas as pd
import numpy as np
import random as rd
import pickle as pkl
from pathlib import Path
import argparse
import itertools


from sklearn.model_selection import KFold


def check_organism(org):
    org_set = {
        "1167": "b",
        "1202": "b",
        "1218": "b",
        "1219": "b",
        "1234": "b",
        "1273": "b",
        "1298": "b",
        "1339": "b",
        "11285": "b",
        "11457": "b",
        "12001": "b",
        "12048": "b",
        "12701": "b",
        "12832": "b",
        "B1": "b",
        "B8": "b",
        "B18": "b",
        "C5": "b",
        "E.ce": "p",
        "E.co": "p",
        "S.en": "p",
        "S.au": "p",
    }
    if org not in org_set.keys():
        raise ValueError(
            f"Organism {org} not in database. Must be one of {org_set.keys()}."
        )
    else:
        return org_set[org]


def hold_out_set(org1, path_to_method_df, org2=None):
    map_column = {"b": "Bacillus", "p": "Pathogene"}

    org1_type = check_organism(org1)
    if org2 is not None:
        org2_type = check_organism(org2)
        assert org1_type != org2_type, (
            f"An interaction cannot be between organisms of same type '{map_column[org2_type]}'."
        )

    path = Path(path_to_method_df).resolve()
    df = pd.read_csv(path)

    if org2 is None:
        # When evaluating for a new organism, we remove it from training
        mask = df[map_column[org1_type]] == org1
        idx_train = df[~mask].index
        idx_test = df[mask].index
    else:
        # When evaluating for a new interaction, we remove both organisms from training
        train_mask = (df[map_column[org1_type]] == org1) | (
            df[map_column[org2_type]] == org2
        )
        test_mask = (df[map_column[org1_type]] == org1) & (
            df[map_column[org2_type]] == org2
        )
        idx_train = df[~train_mask].index
        idx_test = df[test_mask].index

    ho_name = org1 if org2 is None else f"{org1}_x_{org2}"
    # return ho_name, {ho_name:mask}
    return ho_name, {ho_name: (idx_train, idx_test)}


def classical_cross_val(path_to_method_df, cv=5, shuffle=True, random_state=62):
    path = Path(path_to_method_df).resolve()
    df = pd.read_csv(path)

    ho_sets = {}
    if "combinatoric" in path_to_method_df:
        b_unique_ids = list(np.unique(df["B_sample_ID"]))
        print(len(b_unique_ids))
        p_unique_ids = list(np.unique(df["P_sample_ID"]))
        possibilities = [(bid, pid) for pid in p_unique_ids for bid in b_unique_ids]
        # b_possible = {b:list(np.unique(df[df["Bacillus"] == b]["B_sample_ID"])) for b in np.unique(df["Bacillus"])}
        # p_possible = {p:list(np.unique(df[df["Pathogene"] == p]["P_sample_ID"])) for p in np.unique(df["Pathogene"])}

        # b_memory = []
        # p_memory = []
        test_size = len(possibilities) // cv
        rd.seed(random_state)
        for i in range(cv):
            fold_samples = rd.sample(possibilities, test_size)
            df["sample_ID_tuple"] = list(
                map(tuple, df[["B_sample_ID", "P_sample_ID"]].values)
            )
            mask = df["sample_ID_tuple"].isin(fold_samples)
            # fold_b_samples = [idx for B in b_possible.keys() for idx in rd.sample(list(set(b_possible[B]).difference(set(b_memory))), len(b_possible[B]) // cv)]
            # print(set(fold_b_samples).intersection(set(b_memory)))
            # b_memory += fold_b_samples
            # fold_p_samples = [idx for P in p_possible.keys() for idx in rd.sample(list(set(p_possible[P]).difference(set(p_memory))), len(p_possible[P]) // cv)]
            # p_memory += fold_p_samples
            # mask = df["B_sample_ID"].isin(fold_b_samples) | df["P_sample_ID"].isin(fold_p_samples)
            # print(np.sum(mask))
            idx_train = df[~mask].index
            idx_test = df[mask].index
            ho_sets[f"fold_{i}"] = (idx_train, idx_test)
    else:
        splitter = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        for i, (_, test_set) in enumerate(splitter.split(df)):
            mask = df.index.isin(test_set)
            idx_train = df[~mask].index
            idx_test = df[mask].index
            ho_sets[f"fold_{i}"] = (idx_train, idx_test)
    return ho_sets


def to_name(t):
    org1, org2 = t
    if org2 is not None:
        return f"{org1}_x_{org2}"
    else:
        return org1


def all_possible_hold_outs(return_names=False):
    org_set = {
        "1167": "b",
        "1202": "b",
        "1218": "b",
        "1219": "b",
        "1234": "b",
        "1273": "b",
        "1298": "b",
        "1339": "b",
        "11285": "b",
        "11457": "b",
        "12001": "b",
        "12048": "b",
        "12701": "b",
        "12832": "b",
        "B1": "b",
        "B8": "b",
        "B18": "b",
        "C5": "b",
        "E.ce": "p",
        "E.co": "p",
        "S.en": "p",
        "S.au": "p",
    }
    L = [(key, None) for key in org_set.keys()]
    L += [
        (org1, org2)
        for org1 in org_set.keys()
        if org_set[org1] == "b"
        for org2 in org_set.keys()
        if org_set[org2] == "p"
    ]
    if return_names:
        return [to_name(pair) for pair in L]
    return L


def create_all_hold_outs(path_to_method_df, save=False, save_path=None):
    hold_outs_list = all_possible_hold_outs()

    hold_out_sets = {}
    for org1, org2 in hold_outs_list:
        _, ho_dict = hold_out_set(org1, path_to_method_df, org2)
        hold_out_sets.update(ho_dict)
    if save:
        assert save_path is not None, (
            "when save equals True you must provide a save_path."
        )
        with open(save_path, "wb") as f:
            pkl.dump(hold_out_sets, f)
    return hold_out_sets


def get_hold_out_sets(method, ho_folder_path="Data/Datasets/", suffix="_hold_outs.pkl"):
    if method not in ["avg", "random", "combinatoric"]:
        raise ValueError(
            f"method argument must be one of ['avg', 'random', 'combinatoric'], got {method}"
        )

    with open(ho_folder_path + method + suffix, "rb") as f:
        hold_out_sets = pkl.load(f)
    return hold_out_sets


def get_train_test_split(
    ho_name,
    method_df,
    ho_sets,
    target=["Score"],
    remove_cols=["Unnamed: 0"],
    shuffle=False,
    random_state=62,
):
    # hold_out_mask = ho_sets[ho_name]
    train_idx, test_idx = ho_sets[ho_name]
    features = [
        col
        for col in method_df.columns
        if (col not in target) and (col not in remove_cols)
    ]

    # Return X_train, X_test, y_train, y_test
    X_train = method_df.loc[train_idx][features]
    X_test = method_df.loc[test_idx][features]

    y_train = method_df.loc[train_idx][target]
    y_test = method_df.loc[test_idx][target]
    # X_train = method_df[~hold_out_mask][features]
    # X_test = method_df[hold_out_mask][features]

    # y_train = method_df[~hold_out_mask][target]
    # y_test = method_df[hold_out_mask][target]
    if shuffle:
        X_train = X_train.sample(frac=1, random_state=random_state, ignore_index=False)
        y_train = y_train.loc[X_train.index]

    return X_train, X_test, y_train, y_test


import pandas as pd
import numpy as np
import itertools
from datasets import get_train_test_split  # Assumes this function is defined elsewhere


def make_products(df, cols):
    """
    For each unique pair of columns in 'cols', compute their product.
    Returns a DataFrame with the new columns named 'prod_<col1>_<col2>'.
    """
    new_features = pd.DataFrame(index=df.index)
    for col1, col2 in itertools.combinations(cols, 2):
        new_col = f"prod_{col1}_{col2}"
        new_features[new_col] = df[col1] * df[col2]
    return new_features


def make_ratios(df, cols, eps=1e-4):
    """
    For each unique pair of columns in 'cols', compute the ratio col1 / (col2 + eps).
    Returns a DataFrame with new columns named 'ratio_<col1>_<col2>'.
    Only one quotient is computed per pair.
    """
    new_features = pd.DataFrame(index=df.index)
    for col1, col2 in itertools.combinations(cols, 2):
        new_col = f"ratio_{col1}_{col2}"
        new_features[new_col] = df[col1] / (df[col2] + eps)
    return new_features


def make_power(df, cols, orders=[2, 3]):
    """
    For each column in 'cols' and each exponent in 'orders',
    compute the column raised to that power.
    Returns a DataFrame with new columns named 'power_<col>_<order>'.
    """
    new_features = pd.DataFrame(index=df.index)
    for col in cols:
        for order in orders:
            new_col = f"power_{col}_{order}"
            new_features[new_col] = df[col] ** order
    return new_features


def make_density(df, eps=1e-4):
    """
    Computes biofilm density as Height/Volume for both Bacillus and Pathogen.
    Expects the following columns to be present:
      - Bacillus: 'B_Biofilm_Height', 'B_Biofilm_Volume'
      - Pathogen:  'P_Biofilm_Height', 'P_Biofilm_Volume'
    Returns a DataFrame with new columns 'B_Biofilm_Density' and 'P_Biofilm_Density'
    (if the corresponding source columns exist).
    """
    new_features = pd.DataFrame(index=df.index)
    if "B_Biofilm_Height" in df.columns and "B_Biofilm_Volume" in df.columns:
        new_features["B_Biofilm_Density"] = df["B_Biofilm_Height"] / (
            df["B_Biofilm_Volume"] + eps
        )
    if "P_Biofilm_Height" in df.columns and "P_Biofilm_Volume" in df.columns:
        new_features["P_Biofilm_Density"] = df["P_Biofilm_Height"] / (
            df["P_Biofilm_Volume"] + eps
        )
    return new_features


def make_fake_score(df, eps=1e-4):
    """
    Computes a fake score defined as: 1 - (Height^3 / Volume)
    for both Bacillus and Pathogen, based on:
      - Bacillus: 'B_Biofilm_Height' and 'B_Biofilm_Volume'
      - Pathogen:  'P_Biofilm_Height' and 'P_Biofilm_Volume'
    Returns a DataFrame with new columns 'B_Fake_Score' and 'P_Fake_Score'
    (if the corresponding source columns exist).
    """
    new_features = pd.DataFrame(index=df.index)
    if "B_Biofilm_Height" in df.columns and "B_Biofilm_Volume" in df.columns:
        new_features["B_Fake_Score"] = 1 - (df["B_Biofilm_Height"] ** 3) / (
            df["B_Biofilm_Volume"] + eps
        )
    if "P_Biofilm_Height" in df.columns and "P_Biofilm_Volume" in df.columns:
        new_features["P_Fake_Score"] = 1 - (df["P_Biofilm_Height"] ** 3) / (
            df["P_Biofilm_Volume"] + eps
        )
    return new_features


def get_feature_engineered_dataset(
    ho_name,
    method_df,
    ho_sets,
    cols_prod=[None],
    cols_ratio=[None],
    cols_pow=[None],
    eps=1e-4,
    target=["Score"],
    remove_cols=["Unnamed: 0"],
    shuffle=False,
    random_state=62,
):
    """
    Splits the dataset using get_train_test_split and applies feature engineering.
    Each transformation (products, ratios, and powers) is applied only if the corresponding
    column list is not [None]. Density and fake score features are always computed (if the
    required columns are present).

    Returns:
        X_train_fe, X_test_fe, y_train, y_test
    """
    # Obtain the training and test splits.
    X_train, X_test, y_train, y_test = get_train_test_split(
        ho_name,
        method_df,
        ho_sets,
        target=target,
        remove_cols=remove_cols,
        shuffle=shuffle,
        random_state=random_state,
    )

    new_train_features = []
    new_test_features = []

    # Product features
    if cols_prod != [None]:
        prod_train = make_products(X_train, cols_prod)
        prod_test = make_products(X_test, cols_prod)
        new_train_features.append(prod_train)
        new_test_features.append(prod_test)

    # Ratio features
    if cols_ratio != [None]:
        ratio_train = make_ratios(X_train, cols_ratio, eps)
        ratio_test = make_ratios(X_test, cols_ratio, eps)
        new_train_features.append(ratio_train)
        new_test_features.append(ratio_test)

    # Power features
    if cols_pow != [None]:
        power_train = make_power(X_train, cols_pow)
        power_test = make_power(X_test, cols_pow)
        new_train_features.append(power_train)
        new_test_features.append(power_test)

    # Density features
    density_train = make_density(X_train)
    density_test = make_density(X_test)
    new_train_features.append(density_train)
    new_test_features.append(density_test)

    # Fake score features
    fake_score_train = make_fake_score(X_train)
    fake_score_test = make_fake_score(X_test)
    new_train_features.append(fake_score_train)
    new_test_features.append(fake_score_test)

    # Concatenate original features with newly engineered features
    X_train_fe = pd.concat([X_train] + new_train_features, axis=1)
    X_test_fe = pd.concat([X_test] + new_test_features, axis=1)

    return X_train_fe, X_test_fe, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        default=["avg", "random", "combinatoric"],
        nargs="+",
        help="whitespace separated list of the methods to build datasets (avg, random, combinatoric)",
    )
    parser.add_argument(
        "--mode",
        nargs="?",
        default=2,
        type=int,
        help="Specify the type of dataset to build (0=cv or 1=homolgy controled, 2=both). default is 3",
    )

    args = parser.parse_args()
    methods_list = args.methods
    print(methods_list)
    mode = args.mode

    if "avg" in methods_list and mode in [1, 2]:
        print("Creating hold out sets for method 'avg'..")
        t0 = time.time()
        create_all_hold_outs(
            "Data/Datasets/avg_COI.csv",
            save=True,
            save_path="Data/Datasets/avg_hold_outs.pkl",
        )
        print(f"It took {time.time() - t0} seconds.")

    if "random" in methods_list and mode in [1, 2]:
        print("\nCreating hold out sets for method 'random'..")
        t0 = time.time()
        create_all_hold_outs(
            "Data/Datasets/random_COI.csv",
            save=True,
            save_path="Data/Datasets/random_hold_outs.pkl",
        )
        print(f"It took {time.time() - t0} seconds.")

    if "combinatoric" in methods_list and mode in [1, 2]:
        print("\nCreating hold out sets for method 'combinatoric'..")
        t0 = time.time()
        create_all_hold_outs(
            "Data/Datasets/combinatoric_COI.csv",
            save=True,
            save_path="Data/Datasets/combinatoric_hold_outs.pkl",
        )
        print(f"It took {time.time() - t0} seconds.")

    if "avg" in methods_list and mode in [0, 2]:
        print("\nCreating cross_val sets for method 'avg'..")
        t0 = time.time()
        ho_sets = classical_cross_val("Data/Datasets/avg_COI.csv")
        with open("Data/Datasets/avg_cv.pkl", "wb") as f:
            pkl.dump(ho_sets, f)
        print(f"It took {time.time() - t0} seconds.")

    if "random" in methods_list and mode in [0, 2]:
        print("\nCreating cross_val sets for method 'random'..")
        t0 = time.time()
        ho_sets = classical_cross_val("Data/Datasets/random_COI.csv")
        with open("Data/Datasets/random_cv.pkl", "wb") as f:
            pkl.dump(ho_sets, f)
        print(f"It took {time.time() - t0} seconds.")

    if "combinatoric" in methods_list and mode in [0, 2]:
        print("\nCreating cross_val sets for method 'combinatoric'..")
        t0 = time.time()
        ho_sets = classical_cross_val("Data/Datasets/combinatoric_COI.csv")
        with open("Data/Datasets/combinatoric_cv.pkl", "wb") as f:
            pkl.dump(ho_sets, f)
        print(f"It took {time.time() - t0} seconds.")
