# LatticeQuant

## Overview

This repository is designed to accompany ongoing research described in [1], which builds upon the work in [2]. The primary focus of this project is to present and demonstrate lattice quantization with successive refinement, allowing a wide variety of quantization results using compact lookup tables (LUTs). The central implementation for this functionality resides in the `HierarchicalNestedLatticeQuantizer`.

## Core Features

- Hierarchical Nested Lattice Quantizer: Implements a lattice quantization approach that supports successive refinement, enabling a broad range of results with minimal LUT sizes. This is the main implementation, designed to showcase the flexibility and efficiency of the method.

- Nested Lattice Quantizer: A "classic" reference quantizer for comparison purposes, implementing standard Voronoi code quantization.

- Closest Point Algorithms: Algorithms for finding the closest lattice points for well known lattices such as $D_n$, $A_2$ and $E_8$. Algorithms are from [3].

## References

[1] : TBA

[2] : TBA

[3] : J. Conway and N. Sloane, "Fast quantizing and decoding and algorithms for lattice quantizers and codes," in IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 227-232, March 1982, doi: 10.1109/TIT.1982.1056484.

For a full list of references, please see the "References" section of our published work.

## Contributions
Feel free to open issues or submit pull requests for any suggestions or improvements!

