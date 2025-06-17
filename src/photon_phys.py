# V to eV conversion function
def v_to_ev(voltage):
    """

    """
    return voltage  # 1V = 1eV（電子1個あたり）

# wavelength of photon to corresponding eV conversion function
def wavelength_to_ev(wavelength_nm):
    """
    E[eV] = 1239.841984 / λ[nm]
    """
    h = 6.62607015e-34        # Plank Const [J·s]
    c = 2.99792458e8          # [m/s]
    q = 1.602176634e-19       # [C]
    wavelength_m = wavelength_nm * 1e-9  # nm → m
    energy_J = h * c / wavelength_m      # energy [J]
    energy_eV = energy_J / q             # J → eV
    return energy_eV