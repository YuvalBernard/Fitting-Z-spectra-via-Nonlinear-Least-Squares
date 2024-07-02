import os
import pandas as pd

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
import matplotlib.pyplot as plt

# df = pd.read_excel("data.xlsx")
# offsets = jnp.asarray(df["ppm"].to_numpy().astype(float))
# powers = jnp.asarray(list(df.columns[1:].str.extract(r"(\d+.\d+)", expand=False)), dtype=float)
# data = jnp.asarray(df.to_numpy().astype(float).T[1:])

# @jit
# @partial(jnp.vectorize, excluded=[0,1,3,4,5], signature="()->(k)") # powers
# @partial(jnp.vectorize, excluded=[0,2,3,4,5], signature="()->()") # offsets
# def bloch_mcconnell(model_pars, offset, power, B0, gamma, tp):
#     R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_pars
#     dwa *= B0*gamma
#     dwb *= B0*gamma
#     offset *= B0*gamma
#     power *= gamma
#     ka = k * f
#     M0 = jnp.array([0, 0, 1, 0, 0, f, 1])
#     A = jnp.array([
#         [-(R2a + ka), dwa-offset, 0, k, 0, 0, 0],
#         [offset-dwa, -(R2a + ka), power, 0, k, 0, 0],
#         [0, -power, -(R1a + ka), 0, 0, k, R1a],
#         [ka, 0, 0, -(R2b + k), dwb-offset, 0, 0],
#         [0, ka, 0, offset-dwb, -(R2b + k), power, 0],
#         [0, 0, ka, 0, -power, -(R1b + k), R1b * f],
#         [0, 0, 0, 0, 0, 0, 0]])
#     Z = jnp.matmul(expm(A * tp, max_squarings=18), M0)[2]
#     return Z

# fig, ax = plt.subplots()
# ax.plot(offsets, bloch_mcconnell([.33, 0.67, 0, 5, 50, 50, 0.0099, 3.5], offsets, powers, 7.0, 267.522, 10.0).T)
# ax.set_prop_cycle(None)
# ax.plot(offsets, data.T, '.')

# fig.savefig("test1.png")

print(float("0.0003"))