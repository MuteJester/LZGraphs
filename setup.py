import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='LZGraphs',
    author='Thomas Konstantinovsky',
    author_email='thomaskon90@gmail.com',
    description='An Implementation of LZ76 Based Graphs for Repertoire Representation',
    keywords='Graphs,Immunology,Encoding',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MuteJester/LZGraph',
    download_url='https://github.com/MuteJester/LZGraphs/archive/refs/tags/V1.tar.gz',
    project_urls={
        'Documentation': 'https://github.com/MuteJester/LZGraph',
        'Bug Reports':
        'https://github.com/MuteJester/LZGraph/issues',
        'Source Code': 'https://github.com/MuteJester/LZGraph',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={'': 'LZGraphs'},
    packages=setuptools.find_packages(where='LZGraphs'),
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    # install_requires=['Pillow'],
    extras_require={
        'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },
    # entry_points={
    #     'console_scripts': [  # This can provide executable scripts
    #         'run=LZGraph:main',
    # You can execute `run` in bash to run `main()` in LZGraphs/LZGraph/__init__.py
    #     ],
    # },
)
