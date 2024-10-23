# "A 2-layer model of soil hydrology" (Mahrt & Pan, 1984). 
This repository contains a JAX implementation for computing timeseries of soil moisture and runoff surplus, vectorized over point locations. 

As well, a Numpy implementation is provided as comparison: the JAX version resulting in ~3-4x speed up when run on a CPU, and 10-15x speed up on a GPU. 

Implementation is validated in provided tests by comparing to sample data provided by the reference implementation at: https://github.com/ilyamaclean/ecohydrotools/tree/master/data
