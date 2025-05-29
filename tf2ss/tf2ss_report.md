# Transfer Function to State Space Conversion Comparison

This report compares the poles of state-space realizations obtained using different implementations:

1. Our implementation
2. SLYCOT implementation
3. MATLAB implementation

### Comparison Table of Poles

| System | Our Implementation | SLYCOT Implementation | MATLAB Implementation |
|--------|-------------------|----------------------|----------------------|
| $  \frac{1s + 1}{s^2 + 3s + 2}  $ | -2.0 + 0.0j | -2.0 + 0.0j | -2.0 + 0.0j<br>-1.0 + 0.0j |
| $ \begin{bmatrix} \frac{0s^2 + 0s + 1}{s^2 + 0.4s + 3} & \frac{s^2 + 0s + 0}{s^2 + s + 0} \\ \frac{3s^2 - 1s + 1}{s^2 + 0.4s + 3} & \frac{0s^2 + s + 0}{s^2 + s + 0} \\ \frac{0s^2 + 0s + 1}{s^2 + 0.4s + 3} & \frac{0s^2 + 2s + 0}{s^2 + s + 0} \end{bmatrix} $ | 0.0 +0.0j<br>-0.2+1.7205j<br>-0.2-1.7205j<br>-1.0 +0.0j | -0.2+1.7205j<br>-0.2-1.7205j<br>-0.2+1.7205j<br>-0.2-1.7205j<br>-0.2+1.7205j<br>-0.2-1.7205j<br>-1. +0.j<br>-1. +0.j<br>-1. +0.j  | -0.2 + 1.7205j<br>-0.2 - 1.7205j<br>0.0 + 0.0j<br>-1.0 + 0.0j |
| $ \begin{bmatrix} \frac{s^3 + 6s^2 + 12s + 7}{s^3 + 6s^2 + 11s + 6} & \frac{0s^3 + s^2 + 4s + 3}{s^3 + 6s^2 + 11s + 6} \\ \frac{0s^3 + 0s^2 + s + 1}{s^3 + 6s^2 + 11s + 6} & \frac{s^3 + 8s^2 + 20s + 15}{s^3 + 6s^2 + 11s + 6} \end{bmatrix} $ | -3.0 + 0.0j<br>-2.0 + 0.0j<br>-1.0 + 0.0j | -3.0 + 0.0j<br>-2.0 + 0.0j<br>-1.0 + 0.0j<br>-2.0 + 0.0j<br>-2.0 + 0.0j<br>-3.0 + 0.0j<br>-2.0 + 0.0j | -3.0 + 0.0j<br>-2.0 + 0.0j<br>-1.0 + 0.0j<br>-3.0 + 0.0j<br>-2.0 + 0.0j<br>-1.0 + 0.0j |

### Comparison Table of Transmission Zeros

| System | Our Implementation | SLYCOT Implementation | MATLAB Implementation |
|--------|-------------------|----------------------|----------------------|
| $  \frac{1s + 1}{s^2 + 3s + 2}  $ | [] | [] | -1.0 + 0.0j |
| $ \begin{bmatrix} \frac{0s^2 + 0s + 1}{s^2 + 0.4s + 3} & \frac{s^2 + 0s + 0}{s^2 + s + 0} \\ \frac{3s^2 - 1s + 1}{s^2 + 0.4s + 3} & \frac{0s^2 + s + 0}{s^2 + s + 0} \\ \frac{0s^2 + 0s + 1}{s^2 + 0.4s + 3} & \frac{0s^2 + 2s + 0}{s^2 + s + 0} \end{bmatrix} $ | [] | -2.87e-17 + 0.0j<br>0.167 + 0.553j<br>0.167 - 0.553j | -3.1991e-17 + 0.0j |
| $ \begin{bmatrix} \frac{s^3 + 6s^2 + 12s + 7}{s^3 + 6s^2 + 11s + 6} & \frac{0s^3 + s^2 + 4s + 3}{s^3 + 6s^2 + 11s + 6} \\ \frac{0s^3 + 0s^2 + s + 1}{s^3 + 6s^2 + 11s + 6} & \frac{s^3 + 8s^2 + 20s + 15}{s^3 + 6s^2 + 11s + 6} \end{bmatrix} $ | -1.0 + 0.0j<br>-0.333 + 0.0j | -3.618 + 0.0j<br>-2.5 - 0.866j<br>-2.5 + 0.866j<br>-1.382 + 0.0j | -3.2328 + 0.7926j<br>-3.2328 - 0.7926j<br>-1.5344 + 0.0000j<br>-2.0000 + 0.0000j<br>-3.0000 + 0.0000j<br>-1.0000 + 0.0000j |
