# Enzyme Expression Optimization
Web tool based on Python and Streamlit to quickly optimize protein expression using Scikit-learn.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/aa-schoepfer/eeo/main/eeo.py)

Requirements:

- Streamlit
- Seaborn
- Scikit-learn

Start it with:
`$ streamlit run eeo.py`

Test it with the `example.csv` file!

## Installation with Conda
1. Download miniconda from https://docs.conda.io/en/latest/miniconda.html
1. Open an Anaconda Prompt and go to the EEO folder (`cd <PATH>`)
1. Type `conda env create -f eeo.yml` (This takes a while)
1. Actiavte the enviroment with `conda activate eeo`
1. In the EEO folder type `streamlit run eeo.py`
