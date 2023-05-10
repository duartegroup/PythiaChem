import codecs
import os.path
import datetime
import setuptools
from setuptools import setup, find_packages



install_requires = [

]

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def get_readme(rel_path):
    readme = ''
    for line in read(rel_path).splitlines():
        readme += line
    return readme


# use `pip install .`  to install this package manually
def setup(scm=None):
    packages = find_packages()

    setuptools.setup(
        name='pythia',
        use_scm_version=scm,
        setup_requires=['setuptools_scm'],
        version=get_version("pythia/version.py"),
        #author='',
        #author_email='',
        description=('Pythia: Automatic pipeline for ML model building'),
        long_description_content_type="text/markdown",
        long_description=get_readme('README.md'),
        url="https://github.com/duartegroup/Pythia",
        #python_requires="~=3.8",
        packages=packages,
        #package_data={"pythia/data/": ["*.pdb", "*.smi", "*.smarts"]},
        #entry_points={
        #    'console_scripts': [
        #        'pythia=pythia.pipeline:main']
        #},
        classifiers=[
            "Programming Language :: Python :: 3.10",
        ],
        keywords='Machine learning ',
        #install_requires=install_requires,
        include_package_data=True,
        zip_safe=False,
    )

setup()