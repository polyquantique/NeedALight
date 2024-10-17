# Release 0.1.4

### New Features
* New Module "magnus.py" 
    * New function `Magnus1` which calculates the first order Magnus term.
    * New function `Magnus3_Re` which calculates the real part of the third order Magnus term.
    * New function `Magnus3_Im` which calculates the imaginary part of the third order Magnus term.
    * New function `Magnus1CW_data` which calculates the first order Magnus term for a Continuous Wave pump and dispersion relations obtained from data.
    * New function `Magnus3CW_data_Re` which calculates the real part of the third order Magnus term for a Continuous Wave pump and dispersion relations obtained from data.
    * New function `Magnus3CW_data_Im` which calculates the imaginary part of the third order Magnus term for a Continuous Wave pump and dispersion relations obtained from data.
    * New function `Magnus1CW_fit` which calculates the first order Magnus term for a Continuous Wave pump and dispersion relations obtained from fit parameters.
    * New function `Magnus3CW_fit_Re` which calculates the real part of the third order Magnus term for a Continuous Wave pump and dispersion relations obtained from fit parameters.
    * New function `Magnus3CW_fit_Im` which calculates the imaginary part of the third order Magnus term for a Continuous Wave pump and dispersion relations obtained from fit parameters.

### New Documentation
* New notebook `example_magnus.ipynb` which shows how to use the new module for a Gaussian pump and either Gaussian or Sinc PMF.
* New notebook `example_magnus_cw_data.ipynb` which shows how to use the new set of `Magnus#CW_` functions using experimental data.
* New notebook `example_magnus_cw_fit.ipynb` which shows how to the `Magnus#CW_fit_` given a set of fit parameters of a dispersion relation. 

### New Implemntation
* Implemented new test for magnus module.






# Release 0.1.3

### Major Modifications
* Modified how the (z,t) pump envelope is generated in `example_ktspace.ipynb` for the (k,t) code. Now works for arbitrary dispersion.
* Modified (k,t) functions in `propagator.py` to properly account for new method of generating pump envelope.
* Added test to show linear dispersion gives proper solution for pump.

# Release 0.1.2

### New Features
* New function `FtS2` which generates the Fourier Transform of the interaction term in (k,t) space.
* New function `Total_propK` which generates the total Heisenberg Propagator for a given domain in (k,t) space.
* New function `JSAK` which generates the JSA, N-matrix, M-matrix, Ns, and Schmidt number given a Heisenberg Propagator in (k,t) space.

### New Implementations
* Implemented relevant tests for new function.

### New Documentation
* Added `CHANGELOG.md`
* Added `CITATION.cff`
* New notebook `example_ktspace.ipynb` which shows how to use all the new (k,t) functions. Includes the option for both poled and unpoled singlepass (no doublepass, nor self- and cross-phase feature).

### Other Modifications
* Updated `README.md` to include more information on the library as well as how to install easily and funding.
* Modified requirements.txt and setup.py to take into account `custom-poling` changes.

# Release 0.1.1

### New Features
* New function `SXPM_prop` which generates the Heisenberg Propagator in (z,w) space including self- and cross-phase modulation terms for unpoled single pass configuration.

### New Implementations
* Implemented relevant tests for new function.

### New Documentation
* New notebook `example_sxpm.ipynb` which shows how to use the new function `SXPM_prop` for an unpoled singlepass configuration.

# Release 0.1.0
 First release.

 ### New Features
 * New function `Hprop` which calculates the Heisenberg Propagator in (z,w) space for positive and negative poling segments.
 * New function `Total_prog` which given a poling domain, calculates the total Heisenberg Propagator.
 * New function `phases` which removes the free-propagation phases from the Propagator.
 * New function `JSA` which generates the JSA, N-matrix, M-matrix, Ns, and Schmidt number given a Heisenberg Propagator.
 * New function `symplectic_prop` which given a Heisenberg Propagator in the $(a,a^\dagger)$ basis, outputs it in the $(xx,pp)$ basis.

### New Implementations
* Implemented relevant tests for all new functions. 

### New Documentation
* `Cost_Cost_and_implementations.pdf` short note to justify the size of pre-calculated domains by `Hprop`.
* New notebook `example_unpoled.ipynb` which shows how to use the functions for an unpoled singlepass configuration.
* New notebook `example_singlepass.ipynb` which shows how to use the functions for a poled singlepass configuration.
* New notebook `example_doublepass.ipynb` which shows how to use the functions for poled doublepass configuration.
 
