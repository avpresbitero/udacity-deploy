import pickle

CAT_FEAT = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
]


def dump_file(PATH, MODEL):
    """
    Dumps file to path using pickle.
    Parameters
    ----------
    PATH : path of the file to be saved
    MODEL : file to be saved

    Returns
    -------
    None
    """
    with open(PATH, "wb") as output_file:
        pickle.dump(MODEL, output_file)


def load_file(PATH):
    """
    Loads a file using pickle
    Parameters
    ----------
    PATH : path of the file

    Returns
    -------
    None
    """
    with open(PATH, "rb") as file:
        model = pickle.load(file)
    return model
