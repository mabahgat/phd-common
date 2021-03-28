import pkgutil
from unittest import TestLoader, TextTestRunner, TestSuite

from phd_utils import test


def before():
    pass


def get_modules():
    return [modname for _, modname, _ in pkgutil.iter_modules(test.__path__)]


def run_all():
    modules_lst = get_modules()
    test_loader = TestLoader()
    test_suite = test_loader.loadTestsFromNames(modules_lst)
    TextTestRunner(verbosity=2).run(test_suite)


def run_single():
    from phd_utils.test.test_config import TestConfig
    tests_lst = [
        TestConfig('test_get_attribute'),
        TestConfig('test_get_item'),
        TestConfig('test_mix')
    ]
    test_suite = TestSuite()
    test_suite.addTests(tests_lst)
    TextTestRunner(verbosity=2).run(test_suite)


if __name__ == '__main__':
    before()
    run_single()