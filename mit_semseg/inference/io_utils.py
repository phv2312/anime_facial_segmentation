import yaml

"""
io utils
"""
def parse_yml(yml_path):
    data = yaml.load(open(yml_path, 'r'), Loader=yaml.FullLoader)
    return data