import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
from typing import Optional
import pandas as pd
import lmfit
from functools import partial

class ModelParameter:
    def __init__(self, name: str, units: str | None, description: str, vary: Optional[bool]=None,
                init_value: Optional[float]=None, lb: Optional[float]=None, ub: Optional[float]=None, fixed_value: Optional[float]=None, value: Optional[float]=None):
        self.name = name
        self.units = units
        self.vary = vary
        self.description = description
        self.init_value = init_value
        self.lb = lb
        self.ub = ub
        self.fixed_value = fixed_value
        self.value = value # for constants only.

class DummyApp:
    def __init__(self):
        self.df = pd.read_excel("data.xlsx")
        powers = list(self.df.columns[1:].str.extract(r"(\d+.\d+)", expand=False))
        self.offsets = np.asarray(self.df["ppm"].to_numpy(dtype=float))
        self.powers = np.asarray(powers, dtype=float)
        self.data = np.asarray(self.df.to_numpy(dtype=float).T[1:])

        self.B0 = ModelParameter(name="B₀", units="T", description="Static field strength", value=7.4)
        self.gamma = ModelParameter(name="γ", units="10⁶ rad⋅s⁻¹⋅T⁻¹", description="Gyromagnetic ratio", value=103.962)
        self.tp = ModelParameter(name="tₚ", units="s", description="Saturation pulse duration", value=2.0)
        self.R1a = ModelParameter(name="R1a", units="Hz", vary=False, description="Longitudinal relaxation rate of pool a", fixed_value=8.0)
        self.R2a = ModelParameter(name="R2a", units="Hz", vary=False, description="Transverse relaxation rate of pool a", fixed_value=380)
        self.dwa = ModelParameter(name="Δωa", units="ppm", vary=False, description="Larmor frequency of pool a relative to itself.\nShould be zero", fixed_value=0)
        self.R1b = ModelParameter(name="R1b", units="Hz", vary=True, description="Longitudinal relaxation rate of pool b", lb=0.1, ub=100)
        self.R2b = ModelParameter(name="R2b", units="Hz", vary=True, description="Transverse relaxation rate of pool b", lb=1000, ub=100_000)
        self.k = ModelParameter(name="k", units="Hz", vary=True, description="Exchange rate from pool b to pool a", lb=1, ub=500)
        self.f = ModelParameter(name="f", units=None, vary=True, description=("Equilibrium magnetization of pool b relative to pool a."
                                                                         "\nRoughly equivalent to fraction of pool b in solution"), lb=1e-5, ub=0.1)
        self.dwb = ModelParameter(name="Δωb", units="ppm", vary=True, description="Larmor frequency of pool b relative to pool a", lb=-265, ub=-255)
        for p in [self.R1b, self.R2b, self.k, self.f, self.dwb]:
            p.init_value = (p.lb + p.ub)/2

    def fit_spectra(self) -> None:
        params = lmfit.Parameters()
        for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]:
            params.add(name=p.name, value=p.init_value if p.vary else p.fixed_value, vary=p.vary, min=p.lb, max=p.ub)
        
        def residuals(params: lmfit.Parameters, offsets, powers, B0, gamma, tp, data):
            model_pars = np.array([params["R1a"], params["R2a"], params["Δωa"], params["R1b"], params["R2b"], params["k"], params["f"], params["Δωb"]])
            return  (data - bloch_mcconnell(model_pars, offsets, powers, B0, gamma, tp)).flatten()
        
        self.fit = lmfit.minimize(fcn=residuals, params=params, method="COBYLA",
                             args=(self.offsets, self.powers, self.B0.value, self.gamma.value, self.tp.value, self.data))
        self.best_fit_pars = np.asarray(
            [self.fit.params[f"{p.name}"].value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]]
            )
        self.best_fit_spectra = bloch_mcconnell(self.best_fit_pars, self.offsets, self.powers, self.B0.value, self.gamma.value, self.tp.value)

@jit
@partial(jnp.vectorize, excluded=[0,1,3,4,5], signature="()->(k)") # powers
@partial(jnp.vectorize, excluded=[0,2,3,4,5], signature="()->()") # offsets
def bloch_mcconnell(model_pars, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_pars
    dwa *= B0*gamma
    dwb *= B0*gamma
    offset *= B0*gamma
    power *= gamma
    ka = k * f
    M0 = jnp.array([0, 0, 1, 0, 0, f, 1])
    A = jnp.array([
        [-(R2a + ka), dwa-offset, 0, k, 0, 0, 0],
        [offset-dwa, -(R2a + ka), power, 0, k, 0, 0],
        [0, -power, -(R1a + ka), 0, 0, k, R1a],
        [ka, 0, 0, -(R2b + k), dwb-offset, 0, 0],
        [0, ka, 0, offset-dwb, -(R2b + k), power, 0],
        [0, 0, ka, 0, -power, -(R1b + k), R1b * f],
        [0, 0, 0, 0, 0, 0, 0]])
    Z = jnp.matmul(expm(A * tp, max_squarings=18), M0)[2]
    return Z

DummyApp().fit_spectra()