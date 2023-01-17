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
  version = '0.22',
  license='MIT',
  description='An Implementation of LZ76 Based Graphs for Repertoire Representation',
  long_description_content_type="text/markdown",
  long_description=read_md('README.md'),
  author = 'Thomas Konstantinovsky',
  author_email = 'thomaskon90@gmail.com',
  url='https://github.com/MuteJester/LZGraph',
  download_url='https://github.com/MuteJester/LZGraphs/archive/refs/tags/V1.tar.gz',
  keywords = ['Graph Theory','Immunology',
              'analytics,biology','tcell','repertoire','cdr3'],   # Keywords that define your package best
    install_requires=[
        'tqdm',
        'pandas'
    ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
