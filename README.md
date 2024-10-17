# NeedALight
A library that can be used to generate Heisenberg Propagators for waveguided sources, in either frequency-space or momentum-time space, and analyse the temporal mode structure of the light generated.

If you find this library useful in your research please cite our paper, 

Martin Houde and Nicolás Quesada, Waveguided sources of consistent, single-temporal-mode squeezed light: the good, the bad, and the ugly. [AVS Quantum Sci. 5, 011404 (2023)](https://avs.scitation.org/doi/10.1116/5.0133009),

which includes in depth theory concerning the derivation of the Heisenberg Propagators.
### Frequency-Space Solutions
Generates the Heisenberg Propagator for the equations shown in [AVS Quantum Sci. 5, 011404 (2023)](https://avs.scitation.org/doi/10.1116/5.0133009). 

 * Valid for linear dispersions. 
 * Can also include self- and cross-phase modulation terms.
 * Several different examples included in notebooks.
 
### Momentum-Time Solutions
Generates the Heisenberg Propagator for equations of the form given in [Journal of Physics: Photonics 2, 035001 (2020)](https://iopscience.iop.org/article/10.1088/2515-7647/ab87fc/meta) when applied to a non-degenerate system.

 * Valid for all dispersions, including pump. 
 * Example included in notebook.

### Magnus module
Generates first and third order Magnus terms as given in [Phys. Rev. A 90, 063840 (2014)](https://doi.org/10.1103/PhysRevA.90.063840) for pulsed and continuous wave pumps.

* Pulsed code works for Gaussian pump and either Sinc or Gaussian phase-matching function.
* Relies on [Cubature](https://github.com/saullocastro/cubature) package, other functions may not converge.
* CW works for both experimental data or fit parameters. Assumes Sinc phase-matching function.
* Example included in notebook. 


## Installation 
Run the command:

pip install git+https://github.com/polyquantique/NeedALight.git

## Funding

Funding for NeedALight has been supplied by:  

  * Ministère de l'Économie et de l’Innovation du Québec, 
  * Natural Sciences and Engineering Research Council of Canada, 
  * European Union's Horizon Europe Research and Innovation Programme under agreement 101070700 project [MIRAQLS](https://sites.google.com/view/miraqls/).

