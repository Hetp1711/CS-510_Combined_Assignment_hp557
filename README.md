# CS-510 Combined Assignment (hp557)

This repository contains brick puzzle assignments.  The project is implemented in Python and ships with example puzzle layouts and a `run.sh` helper script that wraps the main driver `sbp_Assignment_3.py`.

## Prerequisites

- Python 3.8 or newer (both VS Code and the PSU Tux servers provide compatible versions).
- Git (if you plan to clone the repository yourself).

If you are running on the PSU Tux servers, Python 3 is already installed. On a personal machine or within VS Code, install Python from [python.org](https://www.python.org/downloads/) or your system package manager.

## Repository Structure

- `sbp_Assignment_3.py` – the main solver entry point. This script supports loading puzzle boards, printing legal moves, and searching for solutions.
- `sbp_Assignment-2.py` – the previous assignment submission kept for reference.
- `run.sh` – convenience wrapper that forwards arguments to `sbp_Assignment_3.py`.
- `SBP-*.txt` – puzzle definition files used as input to the solver.

## Running in VS Code

1. **Clone the repository** (or copy it into your VS Code workspace):
  ```bash
    git clone https://github.com/<your-account>/CS-510_Combined_Assignment_hp557.git
    cd CS-510_Combined_Assignment_hp557
  ```
2. **Open the folder** in VS Code (`File → Open Folder…`).
3. When prompted, allow VS Code to create a virtual environment or use your system interpreter (Python 3.8+).
4. **Install optional VS Code extensions**:
   - *Python* extension by Microsoft for IntelliSense and debugging.
5. **Run the solver** from the integrated terminal:
   ```bash
   # Examples
   python sbp_Assignment_3.py print SBP-level0.txt
   python sbp_Assignment_3.py solve SBP-level0.txt
   ```
  You may also invoke the helper script:
    ```bash
   sh run.sh solve SBP-level0.txt
    ```
6. (Optional) **Use the VS Code debugger** by creating a launch configuration that points to `sbp_Assignment_3.py` and passes the same command-line arguments shown above.

## Running on the PSU Tux Servers

1. **Copy or clone the repository** to your home directory on Tux:
   ```bash
   git clone https://github.com/<your-account>/CS-510_Combined_Assignment_hp557.git
   cd CS-510_Combined_Assignment_hp557
   ```
   If you do not have Git access from Tux, upload the files via `scp` or `sftp` instead.
2. **Load the default Python module** (if needed):
   ```bash
   module load python
   ```
3. **Execute the solver** using the system interpreter:
   ```bash
   python3 sbp_Assignment_3.py solve SBP-level0.txt
   ```
  or
  ```bash
   sh run.sh solve SBP-level0.txt
  ```
4. The solver supports multiple commands; run without arguments to see the usage message:
   ```bash
   python3 sbp_Assignment_3.py
   ```

## Common Commands

- `print <puzzle-file>` – Pretty-print the specified puzzle board.
- `moves <puzzle-file>` – List all legal moves from the current board.
- `random <puzzle-file> <steps>` – Perform random legal moves.
- `solve <puzzle-file>` – Attempt to solve the puzzle using informed search.

Refer to the docstrings and inline comments in `sbp_Assignment_3.py` for additional details on the solver algorithms.

## Troubleshooting

- **Module not found / `python` points to Python 2** – Explicitly run `python3` or adjust VS Code's interpreter selection to use Python 3.8+.
- **Permission denied for `run.sh`** – Make the script executable: `chmod +x run.sh`, or invoke it with `sh run.sh ...` as shown above.
- **Input file errors** – Ensure you pass one of the provided `SBP-*.txt` files or follow their format when creating a new puzzle.

## There is competition in assignment 3 module, I need help that how can I optimise this code so that so that the Total Node Search will be quick and fast in level 5,6,7.

Explore the solver by experimenting with different puzzle files, adjusting heuristic strategies inside `sbp_Assignment_3.py`, or extending the CLI with new commands.
