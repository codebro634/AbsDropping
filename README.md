# Time-critical and confidence-based abstraction dropping methods

## Purpose

This is the repository accompanying the paper "Time-critical and confidence-based abstraction
dropping methods" which contains the code to reproduce the experiments and the results of the paper.

## Citing the project

@inproceedings{abs_dropping_schmoecker,
  author    = {Robin Schmöcker, Lennart Kampmann, Alexander Dockhorn},
  title     = {Time-critical and confidence-based abstraction dropping methods},
  booktitle = {Proceedings of the IEEE Conference on Games (COG)},
  year      = {2025},
  address   = {Lisbon, Portugal},
  month     = {August},
  url       = {To be published.}
}

## Abstract

One paradigm of Monte Carlo Tree Search (MCTS)
improvements is to build and use state and/or action
abstractions during the tree search. Non-exact abstractions,
however, introduce an approximation error making convergence
to the optimal action in the abstract space impossible. Hence,
as proposed by Xu et al. , abstraction algorithms should
eventually drop the abstraction. In this paper we propose two novel
dropping schemes, namely OGA-IAAD and OGA-CAD which can
yield clear performance improvements whilst being safe in the
sense that the dropping never causes any notable performance
degradations contrary to Xu’s naive dropping method. OGA-
IAAD is designed for time critical settings while OGA-CAD
is designed to improve the MCTS performance with the same
number of iterations

## Installation

To build the project from source, you will need a C++ compiler supporting the C++20 standard or higher (a lower standard probably works too but we have not tested that). The project
is self-contained and does not require any additional installation.

To compile with [CMake](https://cmake.org/) you need to have CMake installed on your system. A `CMakeLists.txt` file is already provided for configuring the build.

**Steps:**

1. **Clone the repository:**
    ```bash
    git clone https://github.com/codebro634/AbsDropping.git
    cd AbsDropping
    ```

2. **Create a build directory (optional but recommended):**
    ```bash
    mkdir build
    cd build
    ```

3. **Generate build files using CMake:**
    ```bash
    cmake -DCMAKE_CXX_COMPILER=/path/to/your/c++-compiler -DCMAKE_C_COMPILER=/path/to/your/c-compiler ..
    ```

4. **Compile the project:**
    ```bash
    cmake --build .
    ```
   *This will invoke the underlying build system (e.g., `make` or `ninja`) to compile the source code.*

If no errors occur, two compiled binaries `AbsDropDebug` and `AbsDropRelease` should now be available in the `build` directory. The former has been compiled with debug
compiler flags and the latter with aggressive optimization.

## Usage

The program is called with the following arguments:

`--seed`: The seed for the random number of generator. Running the program with the same seed will produce the same results.

`--n`: The number of episodes to run.

`--model`: The abbreviation for the model to use. Example values are
`sa` for SysAdmin, `gol` for Game of Life, `aa` for AcademicAdvising or  `tam` for Tamarisk. The ful list of abbreviations is found in main.cpp in the getModel method.

`--margs`: The arguments for the model. Mostly this is a game map to be specified which can be found in the
`resources` folder.

`--agent`:  Which agent to use. The only options are `mcts` and `oga`.

`--aargs`: The arguments for the agent. A list of required and optional arguments can be found in main.cpp in getAgent.

The following shows an example call of running OGA with 500 Mcts-iterations that drops its abstractions after 100 iterations.

```bash
--seed 42 -n 10  --model gol --margs map=../resources/GameOfLifeMaps/1_Anand.txt  --agent oga --aargs iterations=500 --aargs drop_check_point=0.2
```

