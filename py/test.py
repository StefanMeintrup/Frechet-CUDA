import frechet_cuda as fc
import numpy as np
import pandas as pd
import sys

a = np.array([[0.0],[1.0], [0.0], [1.0]], dtype="d")
b = np.array([[0.0],[0.75], [0.25], [1.0]], dtype="d")
c = np.array([[0.0],[1.0]], dtype="d")


ac = fc.Curve(a)
bc = fc.Curve(b)
cc = fc.Curve(c)

d = fc.Curves()

d.add(ac)
d.add(ac)
d.add(ac)
d.add(ac)
d.add(ac)
d.add(ac)
d.add(ac)
d.add(ac)
d.add(bc)
d.add(bc)
d.add(bc)
d.add(bc)
d.add(cc)
d.add(cc)
d.add(cc)
d.add(cc)


drp = fc.jl_transform(d, 0.5)

distance = fc.compute_distance(ac, bc)
print("FRECHET DISTANCE IS: " + str(distance.value))

distance_rp = fc.compute_distance(drp[0], drp[8])
print("FRECHET DISTANCE IS: " + str(distance_rp.value) + " rp")
