import importlib

DEFAULT_CONFIG = {
    'seed': None,
}


class EnvSpec(object):
    def __init__(self, env_id, entry_point=None):
        self.env_id = env_id
        mod_name, class_name = entry_point.split(':')
        self._entry_point = getattr(importlib.import_module(mod_name), class_name)

    def make(self, config=DEFAULT_CONFIG):
        env = self._entry_point(config)
        return env


class EnvRegistry(object):
    def __init__(self):
        self.env_specs = {}

    def register(self, env_id, entry_point):
        if env_id in self.env_specs:
            raise ValueError('Cannot re-register env_id: {}'.format(env_id))
        self.env_specs[env_id] = EnvSpec(env_id, entry_point)

    def make(self, env_id, config=DEFAULT_CONFIG):
        if env_id not in self.env_specs:
            raise ValueError('Cannot find env_id: {}'.format(env_id))
        return self.env_specs[env_id].make(config)


# Have a global registry
registry = EnvRegistry()


def register(env_id, entry_point):
    return registry.register(env_id, entry_point)


def make(env_id, config={}):
    _config = DEFAULT_CONFIG.copy()
    for key in config:
        _config[key] = config[key]

    return registry.make(env_id, _config)
