from setuptools import setup
try:
    from pypandoc import convert_file
    read_md = lambda f: convert_file(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(
  name = 'LZGraphs',
  packages = ['LZGraphs'],
  version = '0.24',
  license='MIT',
  description='An Implementation of LZ76 Based Graphs for Repertoire Representation',
  long_description_content_type="text/markdown",
  long_description=read_md('README.md'),
  author = 'Thomas Konstantinovsky',
  author_email = 'thomaskon90@gmail.com',
  url='https://github.com/MuteJester/LZGraph',
  download_url='https://github.com/MuteJester/LZGraphs/archive/refs/tags/V2.2.tar.gz',
  keywords = ['Graph Theory','Immunology',
              'analytics,biology','tcell','repertoire','cdr3'],   # Keywords that define your package best
    install_requires=[
        'tqdm',
        'numpy==1.21.5',
        'pandas==1.3.5',
        'networkx==2.8.4',
        'matplotlib==3.5.1',
        'seaborn==0.12.1'
    ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8'
  ],
)
