<p align="center">

[![PyPI version](https://img.shields.io/pypi/v/LZGraphs.svg)](https://pypi.org/project/LZGraphs/)
[![Python versions](https://img.shields.io/pypi/pyversions/LZGraphs.svg)](https://pypi.org/project/LZGraphs/)
[![CI/CD](https://github.com/MuteJester/LZGraphs/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/MuteJester/LZGraphs/actions/workflows/ci-cd.yml)
[![License](https://img.shields.io/github/license/MuteJester/LZGraphs.svg)](https://github.com/MuteJester/LZGraphs/blob/master/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/LZGraphs.svg)](https://pypi.org/project/LZGraphs/)

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

</p>


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/MuteJester/LZGraphs">
    <img src="https://github.com/MuteJester/LZGraphs/blob/master/misc/lzglogo2.png" alt="Logo" width="480" height="330">
  </a>

  <h2 align="center">LZGraphs</h2>

  <p align="center">
    LZ76 Graphs and Applications in Immunology
    <br />
    <a href="https://MuteJester.github.io/LZGraphs/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/MuteJester/LZGraphs/issues">Report Bug</a>
    ·
    <a href="https://github.com/MuteJester/LZGraphs/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project

LZGraphs :dna: is a Python library that implements the methodology presented in the research paper "A Novel Approach to T-Cell Receptor Beta Chain (TCRB) Repertoire Encoding Using Lossless String Compression". 

### Background

The diversity of T-cells is crucial for producing effective receptors that can recognize the pathogens encountered throughout life. A stochastic biological process known as V(D)J recombination accounts for the high diversity of these receptors, making their analysis challenging.

### The LZGraphs Approach

LZGraphs presents a new approach to sequence encoding and analysis, based on the Lempel-Ziv 76 algorithm (LZ-76). By creating a graph-like model, LZGraphs identifies specific sequence features and produces a new encoding approach to an individual’s repertoire. 

This unique repertoire representation allows for various applications, such as:

- Generation probability inference
- Informative feature vector derivation
- Sequence generation
- A new measure for diversity estimation

All of these are obtained without relying on time costly and error-prone alignment steps. 


### Installation

#### General Python Environment

To install LZGraphs in a general Python environment, you can use pip, which is a package manager for Python. Open your terminal and type the following command:

```bash
pip install LZGraphs
```

If you have both Python 2 and Python 3 installed on your machine, and you want to use Python 3, you should use pip3:

```bash
pip3 install LZGraphs
```

#### Jupyter Notebook

If you're using a Jupyter notebook, you can install LZGraphs directly in a code cell. Just type and execute the following command in a new cell:

```python
!pip install LZGraphs
```

The exclamation mark at the beginning is a special Jupyter command that allows you to run terminal commands from within a notebook.

#### Troubleshooting

If you encounter any issues during the installation, make sure that your pip is up-to-date. You can upgrade pip using the following command:

```bash
pip install --upgrade pip
```

Or, for Python 3:

```bash
pip3 install --upgrade pip
```

After upgrading pip, try installing LZGraphs again. If you still encounter issues, please raise an issue in this GitHub repository with a description of the problem and any error messages you received.

---


<!-- USAGE EXAMPLES -->
## Usage

The LZGraphs library is designed to be user-friendly and easy to use. You can get started with it in two main ways:

1. **Read the Documentation**: We have a comprehensive [documentation](https://MuteJester.github.io/LZGraphs/) that provides detailed information about the LZGraph model and its applications. The documentation is divided into several sections to help you understand and use the functions and data structures implemented in this library in the most effective and quick manner. It includes:

    - Installation instructions
    - Tutorials for quick plug-and-play usage
    - Descriptions of miscellaneous, visualization, utilities, and Node Edge Saturation functions
    - Detailed information about the LZGraph Base Class, NDPLZGraph Class, and AAPLZGraph Class

    We recommend starting with the [Tutorials](https://MuteJester.github.io/LZGraphs/tutorials) page for a hands-on introduction to the useful functionality provided by the LZGraph library.

2. **Interactive Jupyter Notebook Guides**: You can also download the `Examples` folder from this repository and follow an interactive Jupyter notebook guide. These guides provide step-by-step instructions on how to use the different models of this repo, making it easy for you to learn by doing.


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/MuteJester/LZGraphs/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open-source community such a powerful place to create new ideas, inspire, and make progress. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT license. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

[Thomas Konstantinovsky]() - thomaskon90@gmail.com

Project Link: [https://github.com/MuteJester/LZGraphs](https://github.com/MuteJester/LZGraphs)





<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/MuteJester/LZGraphs.svg?style=flat-square
[stars-url]: https://github.com/MuteJester/LZGraphs/stargazers
[issues-shield]: https://img.shields.io/github/issues/MuteJester/LZGraphs.svg?style=flat-square
[issues-url]: https://github.com/MuteJester/LZGraphs/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/thomas-konstantinovsky-56230117b/
