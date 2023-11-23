from setuptools import find_packages, setup

REQUIREMENT_PATH = 'requirements.txt'


def get_requirements(requirements_path):
    requirements = []
    with open(requirements_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if '-e.' in requirements:
            requirements.remove('-e.')

    return requirements


setup(
    name='MLProjects',
    version='0.0.1',
    author='HHQ',
    packages=find_packages(),
    install_requires=get_requirements(REQUIREMENT_PATH)
)
