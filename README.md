# Fitting Z-spectra via Nonlinear Least Squares
## Introduction
Z-spetroscopy is a tool in nuclear magnetic resonance (NMR)
that utilizes electromagnetic irradiation to track chemical exchange in solution.

In a nutshell, an [NMR spectrum](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance_spectroscopy) that encapsulates
two different molecules (such as water and [poly-L-lysine](https://en.wikipedia.org/wiki/Polylysine))
is susceptible to their exchange.
Specifically, continuous microwave irradiation about the peak of one species in the spectrum can reduce the peak of the other species.
A Z-spectrum profiles the peak attenuation against the irradiation frequency.
Here is a simple example of a Z-spectrum of amine in water:
![example-Z-spectrum](https://github.com/YuvalBernard/Fitting-Z-spectra-via-Nonlinear-Least-Squares/assets/119048065/c8f1f020-ae81-44e1-9197-a1b343368404)

Attached is a [free review article](https://onlinelibrary.wiley.com/doi/10.1002/mrm.22761) that goes more in depth into the theory.

If the exchanging process can be described as a [first-order reaction](https://en.wikipedia.org/wiki/Rate_equation#First_order), then
the Z-spectrum can be modeled by a set of linear differential equations, called *Bloch-McConnell equations*.

Our lab conducts experiments that generate Z-spectra, then fits them to Bloch-mcConnell equations.
## Project Goal
Take simulated or experimental Z-spectra (supplied by the user)
and fit them to Bloch-McConnell equations, via nonlinear least squares.
## Technical Info
The programm should:
* let the user, though a GUI application:
    - input the path to a csv file that contains Z-spectra that was measured/simulated at multiple irradiation amplitudes
    - input constants such as the irradiation duration and amplitude
    - select which model parameters to fit and which parameters to fix (along with their values)
* fit the spectra to Bloch-McConnell equations and extract the model parameters
* output:
  - a csv file that contains the fit
  - a pdf file with a plot that compares the data and the fit
  - a txt file that summarizes the fit results.
 ## Installing Dependencies
 clone repository and execute the following in the terminal:
```bash
pip install -r requirements.txt
```
## Running the Program
Run
```bash
python bm_fit.py
```
in the terminal
## Testing the Program
Run tests by executing the following in the terminal:
```bash
pytest
```
This project was originally implemented as part of the Python programming course at the Weizmann Institute of Science taught by Gabor Szabo
