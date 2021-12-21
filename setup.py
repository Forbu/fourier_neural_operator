import os
from setuptools import setup
import setuptools

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
    version = "0.8",
    description = ("Library and exemples to use the fourier neural operator"),
    packages=setuptools.find_packages(),
    url='https://zongyi-li.github.io',
    long_description=read('README.md'),
    install_requires=required,
    long_description_content_type='text/markdown',
)
