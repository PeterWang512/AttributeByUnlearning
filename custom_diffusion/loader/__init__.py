from utils import find_and_import_module


def get_dataset(dataset_name, **kwargs):
    # Load dataset dynamically
    module = find_and_import_module("loader", dataset_name)

    # Load dataset from filename
    dataset = module.Dataset(**kwargs)
    return dataset
