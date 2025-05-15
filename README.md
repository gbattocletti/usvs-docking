# SeaCat2 Dynamic Positioning Control

 [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
 ![Python 3.11](https://img.shields.io/badge/python->=3.11-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a collection of dynamic models and controllers for the dynamic positioning of an autonomous Unmanned Surface Vehicle (USV). The target USV is the SeaCAT2, used in the [SeaClear2 project](https://www.seaclear2.eu/).

## Installation
The repository is structured as a python package.
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
The repository is structured as a python package. The project structure is detailed below:
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
│   ├── results/
│   │   └── .pkl, .gif, ...     # saved simulation data (pickle files, gifs...)
│   ├── sim_nl_model.py         # simulation of the nonlinear model without control
│   ├── ...
|   └── sim_visualization.py    # script to generate plots and animations from a sim
│
├── src                         # main package (importable with -e)
│   └── seacat_dp/              # python package name (use for import)
│       ├── __init__.py
│       ├── TODO
│       └── TODO
│
└── tests/                      # Unit/integration tests  
    ├── __init__.py
    ├── TODO
    └── TODO
```

All the relevant simulations are stored in the `scripts/` folder, while all the models, controllers, and required modules are stored under `src/seacat_dp/`

## License

The repository is provided under the GNU GPLv3 License. See the LICENSE file included with
this repository.

---

## Author

[Gianpietro Battocletti](https://www.tudelft.nl/staff/g.battocletti/), PhD Candidate at the [Delft Center for Systems and Control](https://www.tudelft.nl/en/me/about/departments/delft-center-for-systems-and-control/), [Delft University of Technology](https://www.tudelft.nl/en/).

Contact information: [g.battocletti@tudelft.nl]()

Copyright (c) 2025 Gianpietro Battocletti.

---
