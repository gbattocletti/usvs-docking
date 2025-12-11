# SeaCat2 Dynamic Positioning Control

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python 3.11](https://img.shields.io/badge/python->=3.11-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a collection of dynamic models and controllers for the dynamic positioning and cooperative docking of autonomous Uncrewed Surface Vehicles (USV). The target USVs are the SeaCAT2 and SeaDragon, used in the [SeaClear2 project](https://www.seaclear2.eu/).

![](/doc/usvs.png)

## Installation
The repository is structured as a python package.
To install, clone the repository by running:
```sh
git clone https://github.com/gbattocletti/usvs_control.git
cd usvs_control
pip install .
```

In case you want to install also the dependencies used in the development phase you can run instead:
```sh
pip install -e .[dev]
```

## Project Structure
The repository is structured as a python package. The project structure is detailed below:
```
root/  
│  
├── scripts/                    # folder with simulation files and experiments
│   ├── results/
│   │   └── .pkl, .gif, ...     # saved simulation data (pickle data files, gifs...)
│   ├── sim_comparison.py       # compare different controllers for the docking of two USVs
│   ├── sim_docking.py          # simulate the cooperative autonomous docking of two USVs
│   ├── sim_model.py            # simulation of the nonlinear model of a USV without control
│   ├── sim_mpc_sa.py           # single-agent control of a USV for dynamic positioning and reference tracking
|   └── visualize_sim.py        # script to generate plots and animations from an existing pkl data file
│
├── src                         # main package (importable with -e)
│   └── usvs_control/              
│       ├── __init__.py
│       ├── control/            # MPC controllers for dynamic positioning and docking
│       ├── model/              # usv models and parameters
│       ├── utils/              # utilities (I/O management...)
│       └── visualization/      # plot and animation functions
│
├── pyproject.toml              # python project configuration file
├── README.md
├── LICENSE
├── .gitignore                  
└── .pylintrc                   
```

All the relevant simulations files are stored in the `scripts/` folder, while all the models, controllers, and required modules are stored under `src/usvs/`.

---

## Useful links
- Python MSS (T. Fossen) https://github.com/cybergalactic/PythonVehicleSimulator
- MATLAB MSS (T. Fossen) https://github.com/cybergalactic/MSS

## License

The repository is provided under the GNU GPLv3 License. See the LICENSE file included with this repository.

## Author

[Gianpietro Battocletti](https://www.tudelft.nl/staff/g.battocletti/), PhD Candidate at the [Delft Center for Systems and Control](https://www.tudelft.nl/en/me/about/departments/delft-center-for-systems-and-control/), [Delft University of Technology](https://www.tudelft.nl/en/).<br>
Contact information: [g.battocletti@tudelft.nl]().<br>
Copyright (c) 2025 Gianpietro Battocletti.
