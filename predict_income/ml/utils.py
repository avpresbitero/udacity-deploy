import pickle

CAT_FEAT = ["workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",]


def dump_file(PATH, MODEL):
    """

    Parameters
    ----------
    PATH
    MODEL

    Returns
    -------

    """
    with open(PATH, "wb") as output_file:
        pickle.dump(MODEL, output_file)


def load_file(PATH):
    """

    Parameters
    ----------
    PATH

    Returns
    -------

    """
    with open(PATH, "rb") as file:
        model = pickle.load(file)
    return model