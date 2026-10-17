"""Explicit model family dispatch; historical EchoFusion configurations stay valid."""
from .echofusion import build as build_echo
from .wallfusion import build as build_wall


def build(variant, config=None):
    settings = dict(config or {})
    family = settings.pop('family', 'EchoFusion')
    if family == 'EchoFusion':
        return build_echo(variant, settings)
    if family == 'WallFusion':
        return build_wall(variant, settings)
    raise ValueError(f'unknown model family: {family}')


def from_specification(spec):
    expected = {'EchoFusion': 'p8.model.1', 'WallFusion': 'wallfusion.model.1'}
    if expected.get(spec.get('family')) != spec.get('version'):
        raise ValueError('unknown model specification family/version')
    return build(spec['variant'], dict(spec['config'], family=spec['family']))
