import os


def get_project_file(*path):
    return os.path.join(project_dir, *path)


package_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(package_dir)
