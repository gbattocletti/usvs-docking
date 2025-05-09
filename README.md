# USV Dynamic Positioning Control

 [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
 ![Python 3.11](https://img.shields.io/badge/python->=3.11-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a collection of dynamic models and controllers for the dynamic positioning of an autonomous Unmanned Surface Vehicle (USV). The target USV is the SeaCAT2, used in the [SeaClear2 project](https://www.seaclear2.eu/).

## Installation
To install, clone the repository by running:
```sh
git clone https:<REPO URL (TODO)>.git
cd repo-name
```

Create a conda env (or another virtual environment) and install the required dependencies. A 
```sh
conda env create -n <new-env-name> -f environment.yml
```

For development, some additional dependencies can be installed:
```sh

```

Finally, install the package in edit mode to be able to import and use it:
```sh
cd <repo-root>
pip install -e ".[dev]"
```

## Project Structure

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