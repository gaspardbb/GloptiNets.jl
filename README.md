# GloptiNets    

This is the code of the paper [*GloptiNets: Scalable Non-Convex Optimization with Certificates*](https://arxiv.org/abs/2306.14932). 

## Install

This code was executed on Julia 1.8.5. First instantiate the project with `] instantiate`. Then you can run the files in the REPL. CUDA compatible GPU are recommended for interpolation.

## Code structure

I did not build the documentation, but here is an outline of the code structure to get you started.

```
.
├── src
|   ├── GloptiNets.jl        # Training loop in the `interpolate` function and certificate computation
|   ├── psdmodels.jl         # Block PSD models, in Fourier or Chebychev basis
|   ├── polynomials.jl       # Same for polynomials
|   ├── besselsampler.jl     # Sampling from the Bessel distribution
|   └── experimental.jl      # Pieces of code which might turn out being useful
├── scripts                  # For running experiments and be used in the REPL
|   ├── xps                  # Launch large scale experiments to produce the figures of the paper 
|   ├── process_xps          # Process the experiments and produce plots in the REPL
|   ├── gen_random_poly.jl   # Generate the random polynomial for the experiments
|   ├── ...                  # Some scripts which might turn out being useful 
├── data
|   ├── hnorm2               # Polynomials with H norm between 1 and 20 for Fig. 3
|   └── vs_tssos             # Polynomials for Figs. 1 and 2
...
``` 

You can use the `load(::Union{PolyCheby, PolyTrigo}, filename)`, to load the random polynomial and inspect them. See `scripts/tssos_{cheby, trigo}.jl` to see how to convert them to `DynamicPolynomials`. 

### Scripts

- [ ] Optionally remove scripts unrelated to the paper, `{cheby, fourier}_speed.jl`. 

## Possible improvements 

- [ ] If sampling more frequencies from the probability distribution to compute the MoM estimator is necessary, then it would be more interesting to sample the frequencies with `samplesprobas_bycat_wgrid` instead of `samplesprobas_bycat` which uses a hash table.
- [ ] Once [#722](https://github.com/JuliaDiff/ChainRules.jl/issues/722) is solved, merge branch `julia19` which rm `dropdims` 

## Citation

[Gaspard Beugnot, Julien Mairal, & Alessandro Rudi (2023). GloptiNets: Scalable Non-Convex Optimization with Certificates. *arXiv:2306.14932*](https://arxiv.org/abs/2306.14932)