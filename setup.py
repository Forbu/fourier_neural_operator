import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        required = f.read().split('\n')
except:
    required = []

setup(
    name = "fourier_neural_operator",
    version = "0.6",
    description = ("Library and exemples to use the fourier neural operator"),
    packages=['fourier_neural_operator'],
    url='https://zongyi-li.github.io',
    long_description=read('README.md'),
    package_dir={'fourier_neural_operator': 'fourier_neural_operator'},
    install_requires=required,
    long_description_content_type='text/markdown',
)
