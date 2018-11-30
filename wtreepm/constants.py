'''
Constants, conversions, & units, in cgs, unless otherwise noted.

Author[s]: Andrew Wetzel.
'''

# system -----
from __future__ import division
from scipy import constants

# physical constants ----------
grav = constants.gravitational_constant * 1e3    # ~6.67384e-8 {cm ^ 3 / gram / sec ^ 2}
speed_light = constants.speed_of_light * 1e2    # {cm / sec}
proton_mass = constants.m_p * 1e3    # {gram}
boltzmann = constants.k * 1e7    # {erg / K}
electron_volt = constants.electron_volt * 1e7    # {erg}

# astrophysical constants ----------
year = constants.Julian_year    # Julian {sec}
parsec = constants.parsec * 1e2    # ~3.0857e18 {cm}
sun_mass = 1.98892e33    # {gram}
sun_lum = 3.842e33    # {erg}
sun_mag = 4.76    # bolometric (varies with filter but good for sdss r-band)

# conversions ----------
# metric
centi_per_kilo = 1e5
kilo_per_centi = 1 / centi_per_kilo

centi_per_mega = 1e8
mega_per_centi = 1 / centi_per_mega

kilo_per_mega = 1e3
mega_per_kilo = 1 / kilo_per_mega

# mass
gram_per_sun = sun_mass
sun_per_gram = 1 / gram_per_sun

gram_per_proton = proton_mass
proton_per_gram = 1 / proton_mass

# time
sec_per_yr = year
yr_per_sec = 1 / sec_per_yr

sec_per_Gyr = sec_per_yr * 1e9
Gyr_per_sec = 1 / sec_per_Gyr

# length
cm_per_pc = parsec
pc_per_cm = 1 / cm_per_pc

cm_per_kpc = cm_per_pc * 1e3
kpc_per_cm = 1 / cm_per_kpc

cm_per_Mpc = cm_per_pc * 1e6
Mpc_per_cm = 1 / cm_per_Mpc

km_per_pc = cm_per_pc * centi_per_kilo
pc_per_km = 1 / km_per_pc

km_per_kpc = cm_per_pc * 1e-2
kpc_per_km = 1 / km_per_kpc

km_per_Mpc = cm_per_pc * 10
Mpc_per_km = 1 / km_per_Mpc

# energy
erg_per_ev = electron_volt
ev_per_erg = 1 / erg_per_ev

erg_per_kev = erg_per_ev * 1e3
kev_per_erg = 1 / erg_per_ev

kelvin_per_ev = constants.electron_volt / constants.k
ev_per_kelvin = 1 / kelvin_per_ev

# angle
degree_per_radian = 180 / constants.pi
radian_per_degree = 1 / degree_per_radian

arcmin_per_degree = 60
degree_per_arcmin = 1 / arcmin_per_degree

arcsec_per_arcmin = 60
arcmin_per_arcsec = 1 / arcsec_per_arcmin

arcsec_per_degree = arcmin_per_degree * arcsec_per_arcmin
degree_per_arcsec = degree_per_arcmin * arcmin_per_arcsec

arcsec_per_radian = arcsec_per_arcmin * arcmin_per_degree * degree_per_radian
radian_per_arcsec = 1 / arcsec_per_radian

deg2_per_sky = 4 * constants.pi * degree_per_radian ** 2

# cosmological constant parameters
hubble_time = 1 / 100 * Gyr_per_sec * km_per_Mpc    # 1 / H_0 ~ 9.7779 {Gyr / h}
hubble_distance = speed_light / 100 * kilo_per_centi    # c / H_0 ~ 2997.925 {Mpc / h}
hubble_parameter_0 = 100 * Mpc_per_km    # H_0 {h / sec}
# critical density at z = 0:  3 * H_0 ^ 2 / (8 * pi * G) ~ 2.775e11 {M_sun / h / (Mpc / h) ^ 3}
density_critical_0 = (3 * 100 ** 2 / (8 * constants.pi * grav) *
                      centi_per_kilo ** 2 / Mpc_per_cm / sun_mass)

# gravitational constant in various units
# {parsec ^ 3 / M_sun / year ^ 2}
grav_pc_msun_yr = grav * pc_per_cm ** 3 * gram_per_sun * sec_per_yr ** 2
# {kpc ^ 3 / M_sun / year ^ 2}
grav_kpc_msun_yr = grav * kpc_per_cm ** 3 * gram_per_sun * sec_per_yr ** 2
# {Mpc ^ 3 / M_sun / year ^ 2}
grav_Mpc_msun_Gyr = grav * Mpc_per_cm ** 3 * gram_per_sun * sec_per_Gyr ** 2
