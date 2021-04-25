import importlib


def reload_module(module):
    """
    Reloads a module without having restart kernel
    """
    importlib.reload(module)