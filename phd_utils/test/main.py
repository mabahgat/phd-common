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
    from phd_utils.test.test_dataset_creators import TestLiwcCategories, TestLiwcDatasetCreator
    tests_lst = [    
        TestLiwcCategories('test_add_parent_class_if_missing'),
        TestLiwcCategories('test_map_list'),
        TestLiwcCategories('test_map_list2'),
        TestLiwcCategories('test_map_list_parents'),
        TestLiwcCategories('test_keep_lowest_cats_only'),
        TestLiwcDatasetCreator('test_filter'),
        TestLiwcDatasetCreator('test_filter_names'),
        TestLiwcDatasetCreator('test_filer_stopwords'),
        TestLiwcDatasetCreator('test_annotate'),
        TestLiwcDatasetCreator('test_redo_categories'),
        TestLiwcDatasetCreator('test_redo_categories2'),
        TestLiwcDatasetCreator('test_get_raw_annotated'),
        TestLiwcDatasetCreator('test_get_raw_not_annotated'),
        TestLiwcDatasetCreator('test_select_for_test'),
        TestLiwcDatasetCreator('test_select_for_train_top1')
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