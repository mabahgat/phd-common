import unittest
import tempfile # https://www.tutorialspoint.com/generate-temporary-files-and-directories-using-python
import pandas as pd

from phd_utils.dataset_creators import LiwcCategories, LiwcDatasetCreator

class TestLiwcCategories(unittest.TestCase):

    def test_add_parent_class_if_missing(self):
        input_lst = ['i', 'bio', 'health', 'work']
        new_cat_lst = LiwcCategories.add_parent_class_if_missing(input_lst)
        self.assertListEqual(['function', 'pronoun', 'ppron', 'i', 'bio', 'health', 'pconcern', 'work'], new_cat_lst)
    
    def test_map_list(self):
        lc = LiwcCategories(set(('affect', 'bio', 'social')))
        input_lst = ['function', 'pronoun', 'ppron', 'i', 'bio', 'health', 'social', 'male', 'pconcern', 'work']
        new_lst = lc.map_list(input_lst)
        self.assertListEqual(['bio', 'social'], new_lst)
    
    def test_map_list2(self):
        cats = LiwcCategories(set(['social', 'informal', 'relativ', 'function', 'bio', 'percept']))
        ll = [
                ['social', 'family', 'male'], 
                ['informal', 'filler'], 
                ['relativ', 'motion'], 
                ['function', 'pronoun', 'ipron', 'cogproc', 'tentat'], 
                ['function', 'article'], 
                ['social', 'female'],
                ['bio', 'body'],
                ['relativ', 'space'],
                ['relativ', 'space'],
                ['bio', 'ingest'],
                ['function', 'verb', 'timeorient', 'focuspresent', 'relativ', 'motion'],
                ['bio', 'body'],
                ['percept', 'hear', 'pconcern', 'leisure'],
                ['function', 'adj', 'percept', 'feel']
            ]
        for l in ll:
            self.assertNotEqual(len(cats.map_list(l)), 0)
    
    def test_keep_lowest_cats_only(self):
        input_lst = ['function', 'pronoun', 'ppron', 'bio', 'social', 'male', 'pconcern', 'work']
        new_lst = LiwcCategories.keep_lowest_cats_only(input_lst)
        self.assertListEqual(['ppron', 'bio', 'male', 'work'], new_lst)


class TestLiwcDatasetCreator(unittest.TestCase):

    @staticmethod
    def create_empty():
        data_dict = {
            'word':         ['father'   , 'blah'    , 'car'     , 'something'   , 'martin'  , 'a'],
            'diffLikes':    [10         , 2         , 50        , 10            , 2         , 7]
        }
        return LiwcDatasetCreator(pd.DataFrame(data=data_dict))
    
    @staticmethod
    def create_annotated():
        data_dict = {
            'index':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'word': [
                'father',
                'blah',
                'car',
                'something',
                'martin',
                'a',
                'girl',
                'karen',
                'body',
                'street',
                'road',
                'internet',
                'slice',
                'pizza',
                'move',
                'head',
                'music',
                'hot'
            ],
            'diffLikes': [10, 2, 50, 10, 2, 7, 3, 8, -1, 1, 10, 7, -4, 4, 9, 4, -2, 22],
            'liwc': [
                ['social', 'family', 'male'], 
                ['informal', 'filler'], 
                ['relativ', 'motion'], 
                ['function', 'pronoun', 'ipron', 'cogproc', 'tentat'], 
                [], 
                ['function', 'article'], 
                ['social', 'female'],
                [],
                ['bio', 'body'],
                ['relativ', 'space'],
                ['relativ', 'space'],
                [],
                [],
                ['bio', 'ingest'],
                ['function', 'verb', 'timeorient', 'focuspresent', 'relativ', 'motion'],
                ['bio', 'body'],
                ['percept', 'hear', 'pconcern', 'leisure'],
                ['function', 'adj', 'percept', 'feel']
            ]
        }
        df = pd.DataFrame(data=data_dict)
        df.set_index('index')
        return LiwcDatasetCreator(df)
    
    def test_filter(self):
        d = TestLiwcDatasetCreator.create_empty()
        d.filter(set(('car')))
        self.assertTrue('car' not in d.get_raw().word)
    
    def test_filter_names(self):
        d = TestLiwcDatasetCreator.create_empty()
        d.filter_names()
        self.assertTrue('martin' not in d.get_raw().word)
    
    def test_filer_stopwords(self):
        d = TestLiwcDatasetCreator.create_empty()
        d.filter_stopwords()
        self.assertTrue('a' not in d.get_raw().word)
    
    def test_annotate(self):
        d = TestLiwcDatasetCreator.create_empty()
        d.annotate(annotation_type_str='strict')
        raw_df = d.get_raw()
        self.assertTrue('liwc' in raw_df.columns)
    
    def test_redo_categories(self):
        d = TestLiwcDatasetCreator.create_annotated()
        cats = LiwcCategories(set(['social', 'affect']))
        d.redo_categories(with_categories=cats)
        raw_df = d.get_raw()
        self.assertListEqual(['social'], raw_df.iloc[0].liwc)
        self.assertListEqual([] , raw_df.iloc[1].liwc)
    
    def test_redo_categories2(self):
        d = TestLiwcDatasetCreator.create_annotated()
        c1 = len(d.get_raw_annotated())
        cats = LiwcCategories(set(['social', 'informal', 'relativ', 'function', 'bio', 'percept']))
        d.redo_categories(with_categories=cats)
        c2 = len(d.get_raw_annotated())
        self.assertEqual(c1, c2)

    def test_get_raw_annotated(self):
        d = TestLiwcDatasetCreator.create_annotated()
        df = d.get_raw_annotated()
        self.assertTrue(len(df[df.liwc.apply(lambda l: len(l) == 0)]) == 0)
    
    def test_get_raw_not_annotated(self):
        d = TestLiwcDatasetCreator.create_annotated()
        df = d.get_raw_not_annotated()
        self.assertTrue(len(df[df.liwc.apply(lambda l: len(l) == 0)]) != 0)
    
    def test_select_for_test(self):
        d = TestLiwcDatasetCreator.create_annotated()
        c = d.select_for_test(count=11)
        self.assertGreaterEqual(c, 11)
        
    def test_select_for_train_top1(self):
        d = TestLiwcDatasetCreator.create_annotated()
        d.select_for_test(count=11)
        c = d.select_for_train(topN=1)
        self.assertNotEqual(c, 0)
        