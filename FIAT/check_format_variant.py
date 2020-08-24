import re
import warnings


def check_format_variant(variant, degree, element):
    if variant is None:
        variant = "point"
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('Variant of ' + element + ' element will change from point evaluation to integral evaluation.'
                      ' You should project into variant="integral"', DeprecationWarning)

    match = re.match(r"^integral(?:\((\d+)\))?$", variant)
    if match:
        variant = "integral"
        quad_degree, = match.groups()
        quad_degree = int(quad_degree) if quad_degree is not None else 5*(degree + 1)
        if quad_degree < degree + 1:
            raise ValueError("Warning, quadrature degree should be at least %s" % (degree + 1))
    elif variant == "point":
        quad_degree = None
    else:
        raise ValueError('Choose either variant="point" or variant="integral"'
                         'or variant="integral(Quadrature degree)"')

    return (variant, quad_degree)
