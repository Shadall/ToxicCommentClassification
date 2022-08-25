import ast
import yaml
import pathlib
path = pathlib.Path().resolve()


def get_config(path_to_config):
    """Get config

    Args:
        path_to_config: path to config file

    :returns: config
    :rtype: Dict[str, Any]
    """

    with open(path_to_config, mode="r") as file:
        config = yaml.safe_load(file)

    if 'ngram_range' in config['tf-idf']['word']:
        config['tf-idf']['word']['ngram_range'] = ast.literal_eval(
            config['tf-idf']['word']['ngram_range']
        )

    if 'ngram_range' in config['tf-idf']['char']:
        config['tf-idf']['char']['ngram_range'] = ast.literal_eval(
            config['tf-idf']['char']['ngram_range']
        )

    return config
