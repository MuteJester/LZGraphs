from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='LZGraphs',
    version='1.1',
    license='MIT',
    description='An Implementation of LZ76 Based Graphs for Repertoire Representation and Analysis ',
    long_description_content_type="text/markdown",
    long_description=long_description,
    author='Thomas Konstantinovsky',
    author_email='thomaskon90@gmail.com',
    url='https://github.com/MuteJester/LZGraphs',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    download_url='https://github.com/MuteJester/LZGraphs/archive/refs/tags/Beta1.1.0.tar.gz',
    keywords=['Graph Theory', 'Immunology', 'Analytics', 'Biology', 'T-cell', 'Repertoire', 'CDR3'],
    install_requires=requirements,
    python_requires='>=3.8, <4',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
)
