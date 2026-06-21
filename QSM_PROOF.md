# Mathematical Proof of Lossless Quarter-Square Multiplication

This document presents the mathematical proof showing why the **Quarter-Square Multiplication (QSM)** algorithm is exact and lossless when executed on quantized integer values (e.g., $8$-bit integer activations and weights).

---

## 1. Classical Formulation

The Quarter-Square Multiplication algorithm is derived from the identity:
$$(A + B)^2 - (A - B)^2 = (A^2 + 2AB + B^2) - (A^2 - 2AB + B^2) = 4AB$$

Solving for $AB$ yields:
$$AB = \frac{(A + B)^2 - (A - B)^2}{4}$$

In a hardware implementation, the squares are precomputed and stored in a lookup table ($\text{LUT}$):
$$\text{LUT}[n] = n^2$$

This allows multiplication to be evaluated as:
$$AB = \frac{\text{LUT}[A + B] - \text{LUT}[|A - B|]}{4}$$

---

## 2. Parity & Lossless Integer Shift

When mapping this algorithm to integer execution, the division by $4$ is performed as a bitwise right-shift by two bits (`>> 2`). We must prove that the numerator $(\text{LUT}[A + B] - \text{LUT}[|A - B|])$ is always a multiple of $4$, meaning that the right-shift operation **loses zero bits** of precision.

### Theorem: 
For any two integers $A, B \in \mathbb{Z}$, the value $(A + B)^2 - (A - B)^2$ is always divisible by $4$.

### Proof:
Let $S = A + B$ (the sum) and $D = A - B$ (the difference).

#### Case 1: $A$ and $B$ have the same parity (both even or both odd)
* If $A$ and $B$ are both even, their sum $S$ and difference $D$ are both even.
* If $A$ and $B$ are both odd, their sum $S$ (odd + odd = even) and difference $D$ (odd - odd = even) are both even.
* Since $S$ and $D$ are even, we can write:
  $$S = 2k_1, \quad D = 2k_2 \quad \text{for } k_1, k_2 \in \mathbb{Z}$$
* The squares are:
  $$S^2 = 4k_1^2, \quad D^2 = 4k_2^2$$
* Subtracting the squares:
  $$S^2 - D^2 = 4k_1^2 - 4k_2^2 = 4(k_1^2 - k_2^2)$$
* Since $k_1^2 - k_2^2$ is an integer, $S^2 - D^2 \equiv 0 \pmod 4$.

#### Case 2: $A$ and $B$ have different parities (one even, one odd)
* If one is even and one is odd, their sum $S$ (even + odd = odd) and difference $D$ (even - odd = odd) are both odd.
* Since $S$ and $D$ are odd, we can write:
  $$S = 2k_1 + 1, \quad D = 2k_2 + 1 \quad \text{for } k_1, k_2 \in \mathbb{Z}$$
* The squares are:
  $$S^2 = (2k_1 + 1)^2 = 4k_1^2 + 4k_1 + 1$$
  $$D^2 = (2k_2 + 1)^2 = 4k_2^2 + 4k_2 + 1$$
* Subtracting the squares:
  $$S^2 - D^2 = (4k_1^2 + 4k_1 + 1) - (4k_2^2 + 4k_2 + 1) = 4(k_1^2 + k_1 - k_2^2 - k_2)$$
* Since $k_1^2 + k_1 - k_2^2 - k_2$ is an integer, $S^2 - D^2 \equiv 0 \pmod 4$.

### Conclusion:
In all cases, the difference of squares $S^2 - D^2$ is a multiple of $4$. Performing a bitwise shift right by $2$ (`>> 2`) on the integer result of the lookup subtraction is a mathematically exact division with **zero loss of precision**.

---

## 3. Dynamic Range Bounds for Quantized Spaces

For an $8$-bit integer space ($A, B \in [0, 255]$):
* The maximum sum is $S_{\max} = 255 + 255 = 510$.
* The maximum difference is $D_{\max} = |255 - 0| = 255$.
* Therefore, the lookup table $\text{LUT}$ needs to store squares only up to $510$.
* The size of the lookup table is:
  $$\text{Table Size} = 511 \text{ entries}$$
* Since the maximum value of $n^2$ for $n \le 510$ is $510^2 = 260,100$, each entry fits comfortably in a standard $32$-bit unsigned integer (`uint32_t`).
* Total memory footprint of the table:
  $$511 \times 4 \text{ bytes} = 2,044 \text{ bytes} \approx 2 \text{ KB}$$
This fits entirely inside the L1 data cache of any modern CPU/GPU/ASIC, ensuring $O(1)$ single-cycle latency lookup access.
