from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='wqpt-tutorial',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['tests'])
)