
# Files

- MAIN.py
- MAIN.ipynb
- EvolvingTournament.py
- EloScoring.py
- WinScoring.py
- Evolutions.py
- matplotlib_chord.py
- Plotting.py
- StageMatrix.py


# Dependencies

\#pip install <lib>
    
\$pip install --user <lib>

- Required:
    - pandas
    - numpy
    - scipy
    - networkx
    - elo
    - axelrod
    - datetime
    - colorlover
    - seaborn
    - matplotlib
    - plotly
    - itertools
    - random


# Instructions

## Code Structure

- MAIN.ipynb (OR MAIN.py)
    - Import: libraries for importing.
    - Preliminary Tournament: tournament used for the preliminary analysis; sequential run.
        - Setup: variables initialisation and setup.
        - Tournament: play the tournament.
        - Plots and Tables: all plots and tables (heatmaps, ranks, etc).
        - Moran: play Moran process; requires Setup to be run.
    - Evolving Tournament: tournament used for the evolving network; **requires user intervention**.
        - Setup: variables initialisation and setup.
        - Tournament: play the tournament; **rate** and **prob** values must be manually changed.
        - Plots: mean coop rate, and population
        - Network Chart
    - Comparing: reads from files produced by Evolving Tournament
    
