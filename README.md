# Technical note: A validated correction method to quantify organic and inorganic carbon in soils using Rock-Eval® thermal analysis

A Jupyter notebook runnning Python 3 code and the accompanying data needed for implementing a validated correction method of organic and inorganic carbon using Rock-Eval® (RE) thermal analysis.

## The data

The data is a collection of 240 soil samples analyzed using four different machines/analyzers. The file `data/complete_dataset.csv` contains the SIC (Soil Organic Carbon) and SOC (Soil Inorganic Carbon) using different a different suffix for each set of analyses:

1. **CHN data** analyzed at the LAS Arras (National Laboratory of Soil Analysis situated in Arras, France). Suffix `_las`.
2. **RE6 data** analyzed using the RE6 instrument of the ISTeP laboratory situated at Sorbonne University, Paris. Suffix `_istep`.
3. **RE6 data** analyzed using Vinci Technologies' (VT) RE6 instrument, situated in Nanterre, France. Suffix `_vt`.
4. **RE7 data** analuzed using Vinci Technologies' (VT) RE7 instrument. Suffix `_re7`.

## The code

The code was developped using Python 3.8.10 and Jupyter Notebooks running IPython 8.18.1 on Ubuntu 22.04.
Before running the code, the necessary dependencies need to be installed using:

`pip install -r requirements.txt`

The main code is available in the `tictoc.ipynb` notebook, with the majority of the functions implemented in the two python files available in the folder `functions`. 