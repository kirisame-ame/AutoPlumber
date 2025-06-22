from setuptools import find_packages, setup

setup(
    name='autoplumber',
    packages=find_packages(include=['autoplumber', 'autoplumber.*']),
    version='0.1.0',
    description='AutoPlumber: Automated Data Preprocessing for Machine Learning',
    author='Kirisame-ame'
)