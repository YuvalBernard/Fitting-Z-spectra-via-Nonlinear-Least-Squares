from bloch_mcconnell_core_functions import DummyApp
import numpy as np

def test_fit_spectra():
    app = DummyApp()
    app.fit_spectra()
    assert np.allclose(app.best_fit_pars, np.array([8, 380, 0, 53.4916121, 36857.7741, 229.352986, 0.02530176, -255.205701]))