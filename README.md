# discrete-frechet

A dive into the discrete Fréchet distance calculation, from the naïve
approach to high-speed and memory-efficient optimizations.

The discrete Fréchet distance measures the similarity between two 
polygonal curves or polylines. 

This repository contains four different implementations of the 
discrete Fréchet distance calculation.

# Medium Articles

[How long should your dog leash be?](https://medium.com/tblx-insider/how-long-should-your-dog-leash-be-ba5a4e6891fc)

[Fast Discrete Fréchet Distance](https://towardsdatascience.com/fast-discrete-fr%C3%A9chet-distance-d6b422a8fb77)


## Using the Code
The DFD classes live in the `distances` package. They are:
- `DiscreteFrechet`: The classic dynamic programming implementation
using recursion and a NumPy array to store the distance data.
- `LinarDiscreteFrechet`: The linearized implementation of the
previous algorithm avoiding recursion.
- `FastDiscreteFrechetSparse`: Implements the improved algorithm
and uses a sparse array to store the distance data. The sparse
array is implemented as a dictionary.
- `FastDiscreteFrechetMatrix`: Same as above but uses a full-sized
NumPy array to store the distance data. This is the fastest 
implementation of all.

To use the code, select the class to instantiate and initialize it 
with one of the following distance functions:
- `euclidean`: Standard euclidean distance
- `haversine`: Haversine distance on a unit sphere
- `earth_haversine`: Calculates the haversine distance on the
Earth's surface in meters

All distance functions take the point parameters as NumPy arrays
and return the distance as a single float. The haversine distance
functions reverse the parameter indexing order. Instead of (x, y), 
they take (lat, lon). The `hearth_haversine` function takes its
inputs in decimal degrees.

Use the `distance` function of your selected class to calculate
the DFD.
