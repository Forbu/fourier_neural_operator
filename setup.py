import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = "fourier_neural_operator",
    version = "0.1",
    description = ("Library and exemples to use the fourier neural operator"),
    packages=['fourier_neural_operator'],
    url='https://zongyi-li.github.io',
    long_description=read('README.md'),
    package_dir={'fourier_neural_operator': 'fourier_neural_operator'},
    install_requires=required,
    long_description_content_type='text/markdown',
    download_url = "https://github.com/Forbu/fourier_neural_operator/archive/refs/tags/first.tar.gz",
)