# SeaCat2 Dynamic Positioning Control

 [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
 ![Python 3.11](https://img.shields.io/badge/python->=3.11-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a collection of dynamic models and controllers for the dynamic positioning of an autonomous Unmanned Surface Vehicle (USV). The target USV is the SeaCAT2, used in the [SeaClear2 project](https://www.seaclear2.eu/).

## Installation
To install, clone the repository by running:
```sh
git clone https://github.com/gbattocletti/seacat2-dynamic-positioning.git
cd seacat2-dynamic-positioning
pip install .
```

In case you want to install also the dependencies used in the development phase you can run instead:
```sh
pip install -e ".[dev]"
```

## Project Structure
The project is structured as shown below:
```
root/  
│  
├── pyproject.toml              # python project configuration file
├── README.md
├── LICENSE
├── .gitignore                  # git configuration
├── .pylintrc                   # custom settings for linter
│
├── scripts/                    # folder with simulation files and experiments
│
├── src                         # main package (importable with -e)
│   └── seaclear-dp/            # python package name (use for import)
│       ├── __init__.py
│       ├── TODO
│       └── TODO
│
└── tests/                      # Unit/integration tests  
    ├── __init__.py
    ├── TODO
    └── TODO
```