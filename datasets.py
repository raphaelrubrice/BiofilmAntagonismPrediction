import time
import pandas as pd
import numpy as np
import random as rd
import pickle as pkl
from pathlib import Path

from sklearn.model_selection import KFold

def check_organism(org):
    org_set = {"1167":"b", 
               "1202":"b", 
               "1218":"b", 
               "1219":"b", 
               "1234":"b", 
               "1273":"b", 
               "1298":"b", 
               "1339":"b", 
               "11285":"b", 
               "11457":"b", 
               "12001":"b", 
               "12048":"b", 
               "12701":"b", 
               "12832":"b",
               "B1":"b", 
               "B8":"b",
               "B18":"b",
               "C5":"b",
               "E.ce":"p",
               "E.co":"p",
               "S.en":"p",
               "S.au":"p"
               }
    if org not in org_set.keys():
        raise ValueError(f"Organism {org} not in database. Must be one of {org_set.keys()}.")
    else:
        return org_set[org]

def hold_out_set(org1, path_to_method_df, org2=None):
    map_column = {"b":"Bacillus","p":"Pathogene"}

    org1_type = check_organism(org1)
    if org2 is not None:
        org2_type = check_organism(org2)
        assert org1_type != org2_type, f"An interaction cannot be between organisms of same type '{map_column[org2_type]}'."

    path = Path(path_to_method_df).resolve()
    df = pd.read_csv(path)

    if org2 is None:
        mask = df[map_column[org1_type]] == org1
    else:
        mask = (df[map_column[org1_type]] == org1) & (df[map_column[org2_type]] == org2)
    idx_train = df[~mask].index
    idx_test = df[mask].index

    ho_name = org1 if org2 is None else f"{org1}_x_{org2}"
    # return ho_name, {ho_name:mask}
    return ho_name, {ho_name:(idx_train, idx_test)}

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
            df["sample_ID_tuple"] = list(map(tuple, df[["B_sample_ID", "P_sample_ID"]].values))
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
    org_set = {"1167":"b", 
               "1202":"b", 
               "1218":"b", 
               "1219":"b", 
               "1234":"b", 
               "1273":"b", 
               "1298":"b", 
               "1339":"b", 
               "11285":"b", 
               "11457":"b", 
               "12001":"b", 
               "12048":"b", 
               "12701":"b", 
               "12832":"b",
               "B1":"b", 
               "B8":"b",
               "B18":"b",
               "C5":"b",
               "E.ce":"p",
               "E.co":"p",
               "S.en":"p",
               "S.au":"p"
               }
    L = [(key,None) for key in org_set.keys()]
    L += [(org1, org2) for org1 in org_set.keys() if org_set[org1] == "b" for org2 in org_set.keys() if org_set[org2] == "p"]
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
        assert save_path is not None, "when save equals True you must provide a save_path."
        with open(save_path, "wb") as f:
            pkl.dump(hold_out_sets, f)
    return hold_out_sets

def get_hold_out_sets(method, ho_folder_path="Data/Datasets/", suffix="_hold_outs.pkl"):
    if method not in ["avg", "random", "combinatoric"]:
        raise ValueError(f"method argument must be one of ['avg', 'random', 'combinatoric'], got {method}")
    
    with open(ho_folder_path + method + suffix, "rb") as f:
        hold_out_sets = pkl.load(f)
    return hold_out_sets

def get_train_test_split(ho_name, method_df, ho_sets, target=["Score"], remove_cols=['Unnamed: 0'], shuffle=False, random_state=62):
    # hold_out_mask = ho_sets[ho_name]
    train_idx, test_idx = ho_sets[ho_name]
    features = [col for col in method_df.columns if (col not in target) and (col not in remove_cols)]

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

if __name__ == "__main__":
    # ho_name, ho_dict = hold_out_set("B1", "Data/Datasets/avg_COI.csv", "E.ce")

    # ho_mask = ho_dict[ho_name]
    # df = pd.read_csv("Data/Datasets/avg_COI.csv")
    # print(df[~ho_mask].head())
    # print(df[ho_mask].head())
    print("Creating hold out sets for method 'avg'..")
    t0 = time.time()
    create_all_hold_outs("Data/Datasets/avg_COI.csv", save=True, save_path="Data/Datasets/avg_hold_outs.pkl")
    print(f"It took {time.time() - t0} seconds.")

    print("\nCreating hold out sets for method 'random'..")
    t0 = time.time()
    create_all_hold_outs("Data/Datasets/random_COI.csv", save=True, save_path="Data/Datasets/random_hold_outs.pkl")
    print(f"It took {time.time() - t0} seconds.")

    print("\nCreating hold out sets for method 'combinatoric'..")
    t0 = time.time()
    create_all_hold_outs("Data/Datasets/combinatoric_COI.csv", save=True, save_path="Data/Datasets/combinatoric_hold_outs.pkl")
    print(f"It took {time.time() - t0} seconds.")

    print("\nCreating cross_val sets for method 'avg'..")
    t0 = time.time()
    ho_sets = classical_cross_val("Data/Datasets/avg_COI.csv")
    with open("Data/Datasets/avg_cv.pkl", "wb") as f:
            pkl.dump(ho_sets, f)
    print(f"It took {time.time() - t0} seconds.")

    print("\nCreating cross_val sets for method 'random'..")
    t0 = time.time()
    ho_sets = classical_cross_val("Data/Datasets/random_COI.csv")
    with open("Data/Datasets/random_cv.pkl", "wb") as f:
            pkl.dump(ho_sets, f)
    print(f"It took {time.time() - t0} seconds.")
    print("\nCreating cross_val sets for method 'combinatoric'..")
    t0 = time.time()
    ho_sets = classical_cross_val("Data/Datasets/combinatoric_COI.csv")
    with open("Data/Datasets/combinatoric_cv.pkl", "wb") as f:
            pkl.dump(ho_sets, f)
    print(f"It took {time.time() - t0} seconds.")

