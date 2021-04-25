import pkgutil
from unittest import TestLoader, TextTestRunner, TestSuite
import os

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
    from phd_utils.test.test_dataset_v2 import TestUrbanDictWithLiwc
    tests_lst = [    
        TestUrbanDictWithLiwc('test_load')
    ]
    test_suite = TestSuite()
    test_suite.addTests(tests_lst)
    TextTestRunner(verbosity=2).run(test_suite)


if __name__ == '__main__':
    before()
    if 'TEST_TYPE' in os.environ and os.environ['TEST_TYPE'] == 'single':
        run_single()
    else:
        run_all()