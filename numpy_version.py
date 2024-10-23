from functools import cached_property
import numpy as np 


def mahrt_pan_np(
    evapotranspiration : np.ndarray,
    precipitation      : np.ndarray,
    soil_type          : np.ndarray | int,
    axis               : int = 0,
    **kwargs,
) -> dict[str, np.ndarray]:
    """ 'A 2-layer model of soil hydrology', Mahrt and Pan (1984).
    
    Implements the timeseries formulation for a two-layer model of soil
    hydrology at a point location, described by Mahrt and Pan.
    
    Notes
    -----
    Evapotranspiration and precipitation should be at least 1-dimensional 
    arrays (vectors), representing values in time. All parameters must have 
    the same shape and dimensionality if they are arrays. With more than one 
    dimension, the first dimension will represent time by default (but can be
    specified via the axis keyword), and all other dimensions will be 
    vectorized over such that they are treated as independent points.

    Parameters
    ----------
    evapotranspiration : np.ndarray
        Assumed to be the crop reference evapotranspiration.
    precipitation : np.ndarray
        Precipitation in millimeters.
    soil_type
        Type of soil at each point location.
    **kwargs
        All other keywords have default values specified, but may be included
        as static values (constant over all computed series); batched static  
        constants (constant in time, but different per timeseries); dynamic
        (variable over time); or batched dynamic (variable in time, as well
        as different for each timeseries). For possible keywords, see the 
        `configuration` dictionary variable in the code below.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with three jax arrays contained within: {
            'layer_1' : soil water fractions in the top soil layer, 
            'layer_2' : soil water fractions in the bottom soil layer, 
            'surplus' : the direct run-off surplus (mm),
        }. 

    .. Publication
        https://www.researchgate.net/publication/226313395_A_2-layer_model_of_soil_hydrology

    .. Reference implementation
        https://github.com/ilyamaclean/ecohydrotools/blob/master/R/hydromodel.R
        https://rdrr.io/github/ilyamaclean/ecohydrotools/man/hydromodel_time.html

    """

    # Original parameters from ecohydrotools, reordered based on NoahMP:
    #     Label : description of soil type
    #     Smax  : Volumetric water content at saturation (cm^3 / cm^3)
    #     Smin  : Residual water content (cm^3 / cm^3)
    #     alpha : Shape parameter of the van Genuchten mode (cm^-1)
    #     n     : Pore size distribution parameter (dimensionless, > 1)
    #     Ksat  : Saturated hydraulic conductivity (cm / day)
    #     HSG   : USDA Hydrological soil group
    #     oldix : Original index of the table row in ecohydrotools
    soil_parameter_table: np.ndarray = np.array([
     [           'Sand', 0.399, 0.049, 0.0855, 3.02, 330.12,  'A',      1],
     [     'Loamy sand', 0.402, 0.054, 0.0931, 2.09, 223.23,  'A',      2],
     [     'Sandy loam', 0.403, 0.058, 0.0645, 1.77,  80.09,  'A',      3],
     [      'Silt loam', 0.447, 0.067, 0.0171, 1.48,   8.70,  'B',      6],
     [           'Silt', 0.462, 0.035, 0.0154, 1.39,   5.82,  'C',      5],
     [           'Loam', 0.422, 0.074, 0.0322, 1.55,  19.69,  'B',      4],
     ['Sandy clay loam', 0.388, 0.089, 0.0528, 1.44,  24.36,  'C',      7],
     ['Silty clay loam', 0.441, 0.089, 0.0105, 1.30,   1.80,  'D',      9],
     [      'Clay loam', 0.419, 0.091, 0.0213, 1.35,   5.89,  'D',      8],
     [     'Sandy clay', 0.381, 0.103, 0.0312, 1.23,   3.16,  'D',     10],
     [     'Silty clay', 0.368, 0.073, 0.0065, 1.11,   0.67,  'D',     11],
     [           'Clay', 0.394, 0.073, 0.0110, 1.12,   4.48,  'D',     12],
    ], object)# 'Label','Smax','Smin','alpha',  'n', 'Ksat','HSG', 'oldix'

    # Pull out the correct rows for all of the given soil type values
    soil_table_cols = ['Label','Smax','Smin','alpha','n', 'Ksat','HSG','oldix']
    soil_table_rows = np.clip(soil_type, 0, len(soil_parameter_table)-1)
    soil_table_rows = np.atleast_1d(soil_table_rows).astype(int)
    soil_parameters = np.moveaxis(soil_parameter_table[soil_table_rows], -1, 0)
    soil_parameters = dict(zip(soil_table_cols, soil_parameters))
    soil_parameters = {k: np.array(v,float) for k,v in soil_parameters.items()
                        if k not in ['Label', 'HSG']}

    # Set the default parameter configuration, including soil parameters
    configuration = C = soil_parameters | {
        'theta1'  : 0.35,  # initial soil water fraction of top layer
        'theta2'  : 0.35,  # initial soil water fraction of bottom layer
        'surface' : 0,     # surface water depth (mm)
        'z1'      : 5,     # assumed depth of top soil layer 1 (cm)
        'z2'      : 95,    # assumed depth of bottom soil layer 2 (cm)
        'cn'      : 82,    # Runoff curve number
        'cover'   : 0.8,   # Fraction of vegetation cover
        'topratio': 0.5,   # ratio of root water uptake from top layer relative to bottom layer
        'timestep': 3600,  # number of seconds in each time step of model run
        'n2'      : 1.1,   # pore size distribution parameter for controlling ground-water seepage
        'Ksat2'   : 0,     # saturated hydraulic conductivity parameter for controlling ground-water seepage
        'runoff'  : True,  # If True, runoff is assumed to leave the location;
    } | kwargs             # otherwise it is added to rainfall at each timestep
    
    def calculate_timestep(evapot, precip,
        surface, theta1, theta2, z1, z2, cn, cover, topratio,
        timestep, Smin, Smax, alpha, n, n2, Ksat, Ksat2, **_):
        """ Iterate one timestep forward to calculate next theta values """

        # Helpers for bounding array elements within certain ranges
        gt_0 = lambda x: np.clip(x, 0, None)
        lt_1 = lambda x: np.clip(x, None, 0.999)
        in01 = lambda x: np.clip(x, 0.001, 0.999)

        class Genuchten:
            """ Applies the van Genuchten equation to estimate hydraulic 
                parameters of unsaturated soil.
                (van Genuchten 1980 Soil Sci. Soc. of Ameri. Jour. 44:892-898).
            """
            def __init__(self, sw_frac, n, alpha, Ksat, Smin, Smax, timestep):
                self.theta = in01( (sw_frac-Smin)/(Smax-Smin) )
                self.Ksats = (Ksat * timestep) / (24 * 3600)
                self.alpha = alpha * (Smax - Smin)
                self.m = 1 - 1/n

            def _smooth(self, x):
                return (1 - x**(1/self.m)) ** self.m

            @cached_property
            def diffuse(self):
                """ Estimates hydraulic diffusivity. 
                Ksat and the returned value are adjusted so that returned units 
                are in cm per `timestep`. Return value is constrained to be 
                finite by ensuring `sw_frac` is marginally less than `Smax` 
                when saturated, and is assumed to be zero when equal to `Smin`. 
                """
                Ksats = self.Ksats * self.theta ** (0.5 - 1/self.m)
                mnorm = self.m / (1-self.m) * self.alpha
                theta = self._smooth(self.theta)
                return Ksats/mnorm * (theta+1 / theta-2)

            @cached_property
            def conduct(self):
                """ Estimates hydraulic conductivity. 
                Ksat and the returned value are adjusted so that returned units 
                are in cm per `timestep`. 
                """
                Ksats = self.Ksats * self.theta**0.5
                theta = 1 - self._smooth(self.theta)
                return Ksats * theta**2   

        def soil_evapotranspiration(evapot, sw_frac, cover, topratio):
            """ Estimates soil evapotranspiration from the reference ET.
            As the soil water fraction decreases below the saturation value, 
            evaporation from bare soil is usually assumed to continue at the 
            potential rate until the water content decreases below a critical 
            value. Thereafter, evaporation is assumed to decrease linearly with
            decreasing water content, vanishing to zero when water content is 
            low. The `cover` value aportions the ratio of evapotranspiration to
            bare soil evaporation; `topratio` aportions evapotranspiration 
            between the top and bottom layer and depends on the chosen soil 
            layer depths and root depths.
            """
            rat = lt_1(1.08723 * sw_frac ** 0.56007 + 0.1)
            top = evapot * ((1-cover) * rat + cover * topratio)
            bot = evapot * cover * (1-topratio)
            return top, bot

        def runoff(precip, cn, sw_frac, Smin, Smax):
            """ Calculates direct runoff based on USDA Run-off curve number """
            def scn_adjust(cn, sw_frac, Smin, Smax):
                """ Adjustments for antecedent moisture condition """
                theta = (sw_frac - Smin) / (Smax - Smin)
                lower = theta <= 0.25
                upper = theta >= 0.75
                inner = ~(lower | upper)
                cn100 = cn / 100
                return np.where(cn == 100, cn, 100 * (      (cn100 * inner) +
                  ((0.011486 + 0.242536*cn100 + 0.678166*cn100**2) * lower) + 
                  ((0.067300 + 1.610440*cn100 - 0.691140*cn100**2) * upper) ))

            cn2 = scn_adjust(cn, sw_frac, Smin, Smax)
            P = 0.0393701 * precip
            S = (1000 / cn2) - 10
            Ia = 0.2 * S
            Q = (P > Ia) * ((P - Ia) ** 2 / (P - Ia + S))
            return Q / 0.393701

        # Add surplus to precipitation
        theta1 = theta1 + z2 * gt_0(theta2 - Smax)
        precip = precip + z1 * gt_0(theta1 - Smax) * 10 + surface
        thetaM = np.maximum(theta1, theta2)

        # Run model
        G1 = Genuchten(thetaM, n,  alpha, Ksat,  Smin, Smax, timestep)
        G2 = Genuchten(theta2, n2, alpha, Ksat2, Smin, Smax, timestep)
        
        d_coefs = (theta1-theta2) / (z1+z2) * 2
        k_coefs = (theta2!=theta1) * ((theta2<=theta1) + -1*(theta2>theta1))
        surplus = runoff(precip, cn, theta1, Smin, Smax)

        sev1, sev2 = soil_evapotranspiration(evapot, theta1, cover, topratio)
        difference = (precip - surplus) / 10
        
        # Delta due to hydraulic conductivity and diffusivity (cm), which 
        # should do no more than equalize water content of the layers
        limit = abs(theta1-theta2) * z1 * z2 / (z1+z2)
        delta = np.clip(G1.diffuse*d_coefs + G1.conduct*k_coefs, -limit, limit)

        theta1 = theta1 + (difference - (sev1/10) - delta) / z1
        theta2 = theta2 + (G2.conduct - (sev2/10) + delta) / z2

        # Surplus
        theta1  = theta1  + z2 * gt_0(theta2 - Smax) / z1
        surplus = surplus + z1 * gt_0(theta1 - Smax) * 10
        return (np.clip(theta1, Smin, Smax),
                np.clip(theta2, Smin, Smax),
                surplus)

    # Allow equation parameters to also be dynamic in time (which assumes
    # the time axis is the same as for precipitation)
    uses_dt = lambda v: np.shape(v)[:axis+1] == precipitation.shape[:axis+1]
    dynamic = {k: C.pop(k) for k in list(C) if uses_dt(C[k])}

    # Three values: soil water fractions in the top soil layer,  
    # bottom soil layer, and the direct run-off surplus (mm) 
    returns = np.zeros((3,)+precipitation.shape, dtype='float32') * np.nan
    surplus = C['surface']
    offsets = lambda step: (slice(None),) * axis + (step,)
    dimsize = [size for dim,size in enumerate(precipitation.shape) if dim!=axis]

    def broadcast(v):
        """ Add extra dimensions as necessary to allow array broadcasting """
        if np.size(v) > 1:
            assert(np.shape(v)[0] == dimsize[0]), [np.shape(v), dimsize]
            return np.reshape(v, v.shape+(1,)*(len(dimsize)-np.ndim(v)))
        return v
    C = {k: broadcast(v) for k,v in C.items()}

    # Loop over timesteps to sequentially calculate the return values
    for step in map(offsets, range(precipitation.shape[axis])):
        precip_w_surplus = precipitation[step] + (surplus * (not C['runoff']))

        C['theta1'], C['theta2'], surplus = \
            calculate_timestep(evapotranspiration[step], precip_w_surplus, 
                **(C | {k: broadcast(v[step]) for k,v in dynamic.items()}))
        returns[(0,)+step] = C['theta1']
        returns[(1,)+step] = C['theta2']
        returns[(2,)+step] = surplus
    return dict(zip(['layer_1', 'layer_2', 'surplus'], returns))