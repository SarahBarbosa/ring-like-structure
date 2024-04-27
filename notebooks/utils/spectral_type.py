# "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
# http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
# Eric Mamajek
# Version 2019.3.22
#
# Much of the content (but not all) of this table was incorporated
# into Table 5 of Pecaut & Mamajek (2013, ApJS, 208, 9;
# http://adsabs.harvard.edu/abs/2013ApJS..208....9P), so that
# reference should be cited. Parts of this table for A/F/G stars also
# appeared in Table 3 of Pecaut, Mamajek, & Bubar (2012, ApJ 756,
# 154).  For example, Table 5 of Pecaut & Mamajek did not include the
# absolute magnitude and luminosity estimates, nor the properties of
# the LTY dwarfs. 
#
# The author's notes on standard stars and mean parameters for stars
# for each dwarf spectral type can be found in the directory at:
# http://www.pas.rochester.edu/~emamajek/spt/. See further notes and
# caveats after the table. Colors are based on Johnson UBV, Tycho
# BtVt, Cousins RcIc, Gaia DR2 G/Bp/Rp, Sloan izY, 2MASS JHKs,
# and WISE W1/W2/W3/W4 photometry. G-V color is Gaia DR2 G - Johnson V.

import numpy as np

def spectype(Teff: float) -> str:
    spectral_types = {
        "F0V": 7220,
        "F1V": 7030,
        "F2V": 6810,
        "F3V": 6720,
        "F4V": 6640,
        "F5V": 6510,
        "F6V": 6340,
        "F7V": 6240,
        "F8V": 6170,
        "F9V": 6060,
        "F9.5V": 6000,
        "G0V": 5920,
        "G1V": 5880,
        "G2V": 5770,
        "G3V": 5720,
        "G4V": 5680,
        "G5V": 5660,
        "G6V": 5590,
        "G7V": 5530,
        "G8V": 5490,
        "G9V": 5340
    }
    for spectral_type, temperature in spectral_types.items():
        if Teff >= temperature:
            return spectral_type
    return np.NaN