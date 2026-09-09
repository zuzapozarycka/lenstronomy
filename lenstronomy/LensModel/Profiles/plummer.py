"""Class to compute the Plummer 1911 model (https://articles.adsabs.harvard.edu/pdf/1911MNRAS..71..460P)
    This model in 3D is:

.. math::
    \\rho(r) =
    \\frac{3 M_b a^2}
    {4 \\pi (r^2 + a^2)^{5/2}}

where :math:`M_b` is the total bulge mass and :math:`a` is the
Plummer scale radius.

For a Plummer model, the projected half-mass radius is equal to
the scale radius :math:`a`. Therefore, when the bulge half-light
radius is used as the scale radius, we take

.. math::
    a = R_{\\mathrm{half}}.

The lensing calculation uses the projected mass distribution of
the Plummer profile. The projected surface mass density is

.. math::
    \\Sigma(R) =
    \\frac{M_b a^2}
    {\\pi (R^2 + a^2)^2}.
"""

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util import constants

class Plummer(LensProfileBase):

    def __init__(self, light, stellar_mass_bulge, size_bulge_true):
        super().__init__()

        self._light = light
        self._stellar_mass_bulge = stellar_mass_bulge
        self._size_bulge_true = size_bulge_true

    param_names = [
    "a",
    "sigma0",
    "center_x",
    "center_y",
    ]

    lower_limit_default = {
    "sigma0": 0,
    "a": 0,
    "center_x": -100,
    "center_y": -100,
    }

    upper_limit_default = {
    "sigma0": 1e6,
    "a": 100,
    "center_x": 100,
    "center_y": 100,
    }

    @staticmethod
    def density(r, M_b, a):
        """Computes the 3-d density.

        :param r: 3-d radius
        :param M_b: total bulge mass
        :param a: Plummer scale radius
        :return: density at radius r
        """
        rho = (3 * M_b * a**2 / (4 * np.pi * (r**2 + a**2)**(5 / 2)))
        
        return rho

    def density_2d(self, x, y, M_b, a, center_x=0, center_y=0):
        """Projected density along the line of sight at coordinate (x, y).

        :param x: x-coordinate
        :param y: y-coordinate
        :param M_b: total bulge mass
        :param a: Plummer scale radius
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: projected surface mass density
        """
        x_ = x - center_x
        y_ = y - center_y

        # Projected radius from the centre of the profile.
        R = np.sqrt(x_**2 + y_**2)

        # Plummer surface mass density.
        sigma = (
            M_b * a**2
            / (np.pi * (R**2 + a**2)**2)
        )

        return sigma

    @staticmethod
    def mass_3d(r, M_b, a):
        """Mass enclosed within a 3-d sphere of radius r.

        :param r: 3-d radius within the mass is integrated
        :param M_b: total bulge mass
        :param a: Plummer scale radius
        :return: enclosed mass
        """
        mass_3d = (
            M_b
            * r**3
            / (r**2 + a**2)**(3 / 2)
        )

        return mass_3d

    def mass_2d(self, R, M_b, a):
        """Mass enclosed within a projected 2-d radius R.

        :param R: projected radius
        :param M_b: total bulge mass
        :param a: Plummer scale radius
        :return: enclosed projected mass
        """
        mass_2d = M_b * R**2 / (R**2 + a**2)

        return mass_2d

    @staticmethod
    def mass_tot(M_b):
        """Total mass of the Plummer profile.

        :param M_b: total bulge mass
        :return: total mass
        """
        return M_b

    def function(
        self,
        x,
        y,
        sigma0,
        a,
        center_x=0,
        center_y=0,
    ):
        """Lensing potential of the Plummer profile.

        The lensing parameters are:
            sigma0 : lensing normalization
            a      : Plummer scale radius in angular units

        :param x: x-coordinate position [arcsec]
        :param y: y-coordinate position [arcsec]
        :param sigma0: lensing normalization
        :param a: Plummer scale radius [arcsec]
        :param center_x: x-center of the profile [arcsec]
        :param center_y: y-center of the profile [arcsec]
        :return: lensing potential
        """

        x_ = x - center_x
        y_ = y - center_y

        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R, 1e-10)

        # Lensing potential of the Plummer profile.
        potential = sigma0 * 0.5 * np.log(R**2 + a**2)

        return potential

    def derivatives(
        self,
        x,
        y,
        a,
        sigma0,
        center_x=0,
        center_y=0,
    ):
        """Calculate the deflection angles of the Plummer profile.

        The deflection angle is calculated from the enclosed
        projected mass of the Plummer profile.

        :param x: x-coordinate position [arcsec]
        :param y: y-coordinate position [arcsec]
        :param a: Plummer scale radius [arcsec]
        :param sigma0: lensing normalization
        :param center_x: x-center of the profile [arcsec]
        :param center_y: y-center of the profile [arcsec]
        :return: deflection angles alpha_x and alpha_y [arcsec]
        """

        x_ = x - center_x
        y_ = y - center_y

        # Projected angular radius [arcsec]
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R, 1e-10)

        # Radial deflection angle.
        alpha_R = sigma0 * R / (R**2 + a**2)

        # Convert radial deflection into x/y components.
        alpha_x = alpha_R * x_ / R
        alpha_y = alpha_R * y_ / R

        return alpha_x, alpha_y

    def hessian(
        self,
        x,
        y,
        a,
        sigma0,
        center_x=0,
        center_y=0,
    ):
        """Calculate the Hessian terms of the lensing potential analytically.

        :param x: x-coordinate position [arcsec]
        :param y: y-coordinate position [arcsec]
    :   param a: Plummer scale radius [arcsec]
        :param sigma0: lensing normalization
        :param center_x: x-center of the profile [arcsec]
        :param center_y: y-center of the profile [arcsec]
        :return: f_xx, f_xy, f_yx, f_yy
        """

        x_ = x - center_x
        y_ = y - center_y

        R_squared = x_**2 + y_**2

        f_xx = sigma0 * (
            R_squared + a**2 - 2 * x_**2
        ) / (
            (R_squared + a**2)**2
        )

        f_yy = sigma0 * (
            R_squared + a**2 - 2 * y_**2
        ) / (
            (R_squared + a**2)**2
        )

        f_xy = -2 * sigma0 * x_ * y_ / ( (R_squared + a**2)**2 )

        f_yx = f_xy

        return f_xx, f_xy, f_yx, f_yy

    def mass_model_lenstronomy(self, lens_cosmo, spherical=False):
        """Returns the Plummer lens model and its parameters.

        :param lens_cosmo: lenstronomy LensCosmo instance
        :param spherical: whether to use a spherical profile
        :return: lens model list and lens model parameters
        """

        lens_mass_model_list = ["PLUMMER"]

        center_x, center_y = self._light.extended_source_position

        # Stellar mass of the bulge
        M_b = self._stellar_mass_bulge

        # Plummer scale radius in arcsec
        a = self._size_bulge_true

        # Critical surface mass density in M_sun / arcsec^2
        sigma_crit = lens_cosmo.sigma_crit_angle

        # Lensing normalization
        sigma0 = M_b / (np.pi * sigma_crit)

        kwargs_lens_mass = [
            {
                "sigma0": sigma0,
                "a": a,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]

        return lens_mass_model_list, kwargs_lens_mass