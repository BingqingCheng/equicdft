# Mini Lennard--Jones training fields

`mini_lj_nvt.extxyz` contains five complete equilibrium density fields selected
from the project's NVT Lennard--Jones dataset. It is a small, real input for
exercising the training workflow—not a statistically sufficient training set
for a transferable density functional.

All quantities use Lennard--Jones reduced units. Every frame contains a
periodic cubic box of side $8\sigma$, a $16^3$ density grid with spacing
$0.5\sigma$, the time-averaged `density`, and its `V_ext`. The source fields
were generated in NVT simulations under spatially varying periodic Gaussian
external potentials.

One field was selected at each of five temperatures. Together they span a
compact range of mean densities:

| $T^*$ | source frame | $N$ | mean density |
|---:|---:|---:|---:|
| 0.8 | 66 | 96 | 0.1875 |
| 1.0 | 96 | 144 | 0.28125 |
| 1.2 | 114 | 192 | 0.3750 |
| 1.4 | 132 | 240 | 0.46875 |
| 1.6 | 150 | 288 | 0.5625 |

The original source filename and zero-based frame index are retained in each
EXTXYZ header as `example_source_file` and `example_source_frame`.

The default example split uses three complete fields for training and two for
validation. No grid points from one field are distributed across subsets.
