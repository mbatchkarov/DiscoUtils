# coding=utf-8
from glob import glob
import shelve
from unittest import TestCase
import os
from discoutils.tokens import DocumentFeature
from pandas import DataFrame

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from operator import itemgetter
from scipy.sparse import issparse
from discoutils.thesaurus_loader import Thesaurus, Vectors
from discoutils.collections_utils import walk_nonoverlapping_pairs, walk_overlapping_pairs

__author__ = 'mmb28'


@pytest.fixture
def thesaurus_c():
    return Thesaurus.from_tsv('discoutils/tests/resources/exp0-0c.strings',
                              sim_threshold=0,
                              include_self=False,
                              ngram_separator='_')


@pytest.fixture(params=['txt', 'gz', 'hdf'])
def vectors_c(request, tmpdir):
    kind = request.param  # txt, gz or hdf
    v = Vectors.from_tsv('discoutils/tests/resources/exp0-0c.strings', sim_threshold=0, ngram_separator='_')
    assert DocumentFeature.from_string('oversized/J') not in v
    assert len(v) == 5
    return _generate_hdf_gzip_repr(kind, tmpdir, v)


def _generate_hdf_gzip_repr(kind, tmpdir, v):
    if kind == 'txt':
        # just read the plaintext file
        return v
    else:
        outfile = str(tmpdir.join('events.txt'))
        if kind == 'gz':
            v.to_tsv(outfile, gzipped=True)
        if kind == 'hdf':
            v.to_tsv(outfile, dense_hd5=True)
        return Vectors.from_tsv(outfile)


@pytest.fixture(params=['txt', 'gz', 'hdf'])
def overlapping_vectors(request, tmpdir, _overlapping_vectors):
    kind = request.param  # txt, gz or hdf
    return _generate_hdf_gzip_repr(kind, tmpdir, _overlapping_vectors)


@pytest.fixture(params=[True, False])
def _overlapping_vectors(request):
    return Vectors.from_tsv('discoutils/tests/resources/lexical-overlap-vectors.txt',
                            allow_lexical_overlap=request.param)


@pytest.fixture
def thes_without_overlap():
    return Thesaurus.from_tsv('discoutils/tests/resources/lexical-overlap.txt',
                              sim_threshold=0,
                              ngram_separator='_',
                              allow_lexical_overlap=False)


@pytest.fixture
def thes_with_overlap():
    return Thesaurus.from_tsv('discoutils/tests/resources/lexical-overlap.txt',
                              sim_threshold=0,
                              ngram_separator='_',
                              allow_lexical_overlap=True)


def test_nearest_neighbours(vectors_c):
    entries_to_include = ['b/V', 'g/N', 'a/N']
    from sklearn.metrics.pairwise import euclidean_distances

    mat = vectors_c.matrix
    sims_df = DataFrame(euclidean_distances(mat), index=vectors_c.row_names, columns=vectors_c.row_names)
    print('Cosine sims:\n', sims_df)  # all cosine sim in a readable format

    vec_df = DataFrame(mat.A if issparse(mat) else mat,
                       columns=vectors_c.columns, index=vectors_c.row_names)
    assert mat.shape == (5, 5) == (len(vectors_c.row_names), len(vectors_c.columns))
    print('Vectors:\n', vec_df)

    assert len(vectors_c.get_nearest_neighbours('a/N')) == 4 # overlap allowed
    assert len(vectors_c.get_nearest_neighbours('d/J')) == 4
    assert len(vectors_c.get_nearest_neighbours('asdfa')) == 0 # not contained

    vectors_c.init_sims(n_neighbors=1)  # insert all entries
    vectors_c.allow_lexical_overlap = True
    neigh = vectors_c.get_nearest_neighbours('b/V')
    assert len(neigh) == 1
    assert neigh[0][0] == 'a/J_b/N'  # seeking nearest neighbour of something we trained on
    assert abs(neigh[0][1] - 0.223607) < 1e-5
    assert vectors_c.nn._fit_X.shape == (5, 5)

    vectors_c.init_sims(entries_to_include, n_neighbors=1)
    neigh = vectors_c.get_nearest_neighbours('b/V')
    assert len(neigh) == 1
    assert len(neigh) == 1
    assert neigh[0][0] == 'a/N'  # seeking nearest neighbour of something we trained on
    assert abs(neigh[0][1] - 0.387298) < 1e-5
    assert vectors_c.nn._fit_X.shape == (3, 5)

    neigh = vectors_c.get_nearest_neighbours('a/J_b/N')
    assert len(neigh) == 1
    assert neigh[0][0] == 'b/V'
    assert abs(neigh[0][1] - 0.223607) < 1e-5

    vectors_c.init_sims(entries_to_include, n_neighbors=2)
    neigh = vectors_c.get_nearest_neighbours('b/V')
    assert len(neigh) == 2
    assert neigh[0][0] == 'a/N'
    assert neigh[1][0] == 'g/N'
    assert abs(neigh[0][1] - 0.387298) < 1e-5
    assert abs(neigh[1][1] - 1.315295) < 1e-5

    # test lexical overlap
    vectors_c.allow_lexical_overlap = False
    vectors_c.init_sims(entries_to_include, n_neighbors=2)
    neigh = vectors_c.get_nearest_neighbours('b/V')
    assert neigh[0][0] == 'a/N'
    assert abs(neigh[0][1] - 0.387298) < 1e-5
    assert len(neigh) == 2


def test_nearest_neighbours_too_few_neighbours(vectors_c):
    # check that no error is raised when asking for more neighbours than there are
    entries_to_include = ['b/V', 'g/N', 'a/N']
    vectors_c.init_sims(entries_to_include, n_neighbors=99999)
    neigh = vectors_c.get_nearest_neighbours('b/V')
    assert len(neigh) == 2

    # corner case- less neighbours than requested are returned if the pool of neighbour is exhausted
    vectors_c.init_sims(entries_to_include, n_neighbors=3)
    neigh = vectors_c.get_nearest_neighbours('b/V')
    assert len(neigh) == 2


def test_get_nearest_neigh_compare_to_byblo(vectors_c):
    thes = 'discoutils/tests/resources/thesaurus_exp0-0c/test.sims.neighbours.strings'
    if not os.path.exists(thes):
        pytest.skip("The required resources for this test are missing. Please add them.")
    else:
        byblo_thes = Thesaurus.from_tsv(thes)
        vectors_c.init_sims(n_neighbors=4, nn_metric='cosine')

        assert set(vectors_c.keys()) == set(byblo_thes.keys())
        for entry, byblo_neighbours in byblo_thes.items():
            my_neighbours = vectors_c.get_nearest_neighbours(entry)

            for ((word1, sim1), (word2, sim2)) in zip(my_neighbours, byblo_neighbours):
                assert word1 == word2
                # byblo uses cosine similarity, which is 1 - cosine distance.
                # check if the two sum to 1
                assert abs(sim1 + sim2 - 1.) < 1e-5

    # check if the distance to neighbour increases as we go down the list
    for a, b in walk_overlapping_pairs([x[1] for x in my_neighbours]):
        assert a <= b

    for a, b in walk_overlapping_pairs([x[1] for x in byblo_neighbours]):
        assert a >= b

def test_nearest_neighbours_skipping(vectors_c):
    vectors_c.init_sims()
    print(vectors_c.get_nearest_neighbours_linear('b/V'))
    print(vectors_c.get_nearest_neighbours_linear('a/J_b/N'))
    print(vectors_c.get_nearest_neighbours_linear('a/N'))
    print(vectors_c.get_nearest_neighbours_linear('g/N'))
    print(vectors_c.get_nearest_neighbours_linear('d/J'))
    neigh = vectors_c.get_nearest_neighbours_skipping('b/V')
    neigh = [x[0] for x in neigh]
    print(neigh)
    # only four neighbours, as there are a total of 5 entries in the entire thesaurus
    assert neigh == ['a/J_b/N', 'd/J', 'a/N', 'g/N']


def test_similarity_calculation_match(vectors_c):
    """
    Test that the similarity scores returned by get_nearest_neighbours_linear,
    get_nearest_neighbours_skipping and cos_similarity match
    """
    vectors_c.init_sims()
    for method in ['get_nearest_neighbours_linear', 'get_nearest_neighbours_skipping']:
        for neigh, sim in getattr(vectors_c, method)('b/V'):
            assert abs(sim - vectors_c.euclidean_distance(neigh, 'b/V')) < 1e-5


def test_get_vector(vectors_c):
    df1 = DataFrame(vectors_c.matrix.A, columns=vectors_c.columns,
                    index=map(itemgetter(0), sorted(vectors_c.rows.items(), key=itemgetter(1))))
    for entry in thesaurus_c.keys():
        a = vectors_c.get_vector(entry).A.ravel()
        b = df1.loc[entry].values
        assert_array_almost_equal(a, b)


def test_euclidean_distance(vectors_c):
    assert vectors_c.euclidean_distance('a/N', 'g/N') > 0
    assert vectors_c.euclidean_distance('a/N', 'a/N') == 0.0
    assert vectors_c.euclidean_distance('afdsf', 'fad') is None


def test_loading_bigram_thesaurus(thesaurus_c):
    assert len(thesaurus_c) == 5
    assert 'a/J_b/N' in thesaurus_c.keys()
    assert 'messed_up' not in thesaurus_c.keys()


def test_disallow_lexical_overlap(thes_without_overlap):
    # entries wil only overlapping neighbours must be removed
    assert len(thes_without_overlap) == 3
    # check the right number of neighbours are kept
    assert len(thes_without_overlap['monetary/J_screw/N']) == 1
    assert len(thes_without_overlap['daily/J_pais/N']) == 2
    # check the right neighbour is kept
    assert thes_without_overlap['japanese/J_yen/N'][0] == ('daily/J_mark/N', 0.981391)


def test_disallow_lexical_overlap_with_vectors(overlapping_vectors):
    m = overlapping_vectors.matrix
    assert m.shape == (4, 4)

    overlapping_vectors.init_sims()
    neigh = overlapping_vectors.get_nearest_neighbours('daily/J_pais/N')
    if overlapping_vectors.allow_lexical_overlap:
        assert len(neigh) == 3
    else:
        assert neigh == [('spanish/J', 0.0)]


@pytest.mark.parametrize('thes', [thesaurus_c(), thes_with_overlap(), thes_without_overlap()])
def test_from_shelf(thes, tmpdir):
    filename = str(tmpdir.join('test_shelf'))
    thes.to_shelf(filename)
    loaded_thes = Thesaurus.from_shelf_readonly(filename)
    for k, v in thes.items():
        assert k in loaded_thes
        assert v == loaded_thes[k]


def test_allow_lexical_overlap(thes_with_overlap):
    assert len(thes_with_overlap) == 5
    assert len(thes_with_overlap['monetary/J_screw/N']) == 5
    assert len(thes_with_overlap['daily/J_pais/N']) == 5
    assert thes_with_overlap['japanese/J_yen/N'][0] == ('bundesbank/N_yen/N', 1.0)


# todo check this
def _assert_matrix_of_thesaurus_c_is_as_expected(matrix, rows, cols):
    # rows may come in any order
    assert set(rows) == set(['g/N', 'a/N', 'd/J', 'b/V', 'a/J_b/N'])
    # columns must be in alphabetical order
    assert cols == ['a/N', 'b/V', 'd/J', 'g/N', 'x/X']
    # test the vectors for each entry
    expected_matrix = np.array([
        [0.1, 0., 0.2, 0.8, 0.],  # ab
        [0., 0.1, 0.5, 0.3, 0.],  # a
        [0.1, 0., 0.3, 0.6, 0.],  # b
        [0.5, 0.3, 0., 0.7, 0.],  # d
        [0.3, 0.6, 0.7, 0., 0.9]  # g
    ])
    # put the rows in the matrix in the order in which they are in expected_matrix
    matrix_ordered_by_rows = matrix[np.argsort(np.array(rows)), :]
    assert_array_equal(matrix_ordered_by_rows, expected_matrix)

    vec_df = DataFrame(matrix, columns=cols, index=rows)
    from pandas.util.testing import assert_frame_equal

    expected_frame = DataFrame(expected_matrix, index=['a/J_b/N', 'a/N', 'b/V', 'd/J', 'g/N'], columns=cols)
    assert_frame_equal(vec_df.sort(axis=0), expected_frame.sort(axis=0))


def test_to_sparse_matrix(thesaurus_c):
    matrix, cols, rows = thesaurus_c.to_sparse_matrix()
    matrix = matrix.A
    assert matrix.shape == (5, 5)

    _assert_matrix_of_thesaurus_c_is_as_expected(matrix, rows, cols)

    data = np.array([
        [0., 0.1, 0.5, 0.3, 0.],  # a
        [0.1, 0., 0.3, 0.6, 0.],  # b
        [0.3, 0.6, 0.7, 0., 0.9],  # g
        [0.1, 0., 0.2, 0.8, 0.]  # a_n
    ])


def test_to_dissect_core_space(vectors_c):
    """
    :type vectors_c: Thesaurus
    """
    space = vectors_c.to_dissect_core_space()
    matrix = space.cooccurrence_matrix.mat.A
    _assert_matrix_of_thesaurus_c_is_as_expected(matrix, space.id2row, space.id2column)


def test_thesaurus_to_tsv(thesaurus_c, tmpdir):
    """

    :type thesaurus_c: Thesaurus
    :type tmpdir: py.path.local
    """
    # test columns(neighbours) are not reordered by Thesaurus
    filename = str(tmpdir.join('outfile.txt'))
    thesaurus_c.to_tsv(filename)
    t1 = Thesaurus.from_tsv(filename)
    assert t1._obj == thesaurus_c._obj


def test_vectors_to_tsv(vectors_c, tmpdir):
    """

    :type vectors_c: Vectors
    :type tmpdir: py.path.local
    """
    # these are feature vectors, columns(features) can be reordered
    filename = str(tmpdir.join('outfile.txt'))
    vectors_c.to_tsv(filename, gzipped=True)
    from_disk = Vectors.from_tsv(filename)

    if hasattr(vectors_c, 'df'):
        # this is in dense format
        np.testing.assert_array_equal(vectors_c.matrix, from_disk.matrix)
    else:
        # sparse format: can't just assert from_disk == thesaurus_c, because to_tsv may reorder the columns
        for k, v in vectors_c.items():
            assert k in from_disk.keys()
            assert set(v) == set(vectors_c[k])


def test_get_vector(vectors_c):
    assert_array_equal([0.5, 0.3, 0., 0.7, 0.],
                       vectors_c.get_vector('d/J').A.ravel())
    assert_array_equal([0.3, 0.6, 0.7, 0., 0.9],
                       vectors_c.get_vector('g/N').A.ravel())


def test_loading_unordered_feature_lists(tmpdir):
    d = {
        'a/N': [('f1', 1), ('f2', 2), ('f3', 3)],
        'b/N': [('f3', 3), ('f1', 1), ('f2', 2), ],
        'c/N': [('f3', 3), ('f2', 2), ('f1', 1)],
    }  # three identical vectors
    v = Vectors(d)
    filename = str(tmpdir.join('outfile.txt'))
    v.to_tsv(filename)

    v1 = v.from_tsv(filename)
    assert v.columns == v1.columns # rows can be in any order, but columns need to be sorted
    for word in d.keys():
        assert_array_equal(v.get_vector(word).A, v1.get_vector(word).A)


def test_loading_dict_of_dicts():
    d = {
        'monday': {
            'det:the': 23,
            'amod:terrible': 321
        },
        'tuesday': {
            'amod:awful': 231,
            'det:a': 12
        }
    }
    v = Vectors(d)

    v1 = v.from_dict_of_dicts(d)
    assert v.columns == v1.columns
    for word in d.keys():
        assert_array_equal(v.get_vector(word).A, v1.get_vector(word).A)


def test_to_dissect_sparse_files(vectors_c, tmpdir):
    """

    :type vectors_c: Thesaurus
    :type tmpdir: py.path.local
    """
    from composes.semantic_space.space import Space

    prefix = str(tmpdir.join('output'))
    vectors_c.to_dissect_sparse_files(prefix)
    # check that files are there
    for suffix in ['sm', 'rows', 'cols']:
        outfile = '{}.{}'.format(prefix, suffix)
        assert os.path.exists(outfile)
        assert os.path.isfile(outfile)

    # check that reading the files in results in the same matrix
    space = Space.build(data="{}.sm".format(prefix),
                        rows="{}.rows".format(prefix),
                        cols="{}.cols".format(prefix),
                        format="sm")

    matrix, rows, cols = space.cooccurrence_matrix.mat, space.id2row, space.id2column
    exp_matrix, exp_cols, exp_rows = vectors_c.to_sparse_matrix()

    assert exp_cols == cols
    assert exp_rows == rows
    assert_array_equal(exp_matrix.A, matrix.A)
    _assert_matrix_of_thesaurus_c_is_as_expected(matrix.A, rows, cols)
    _assert_matrix_of_thesaurus_c_is_as_expected(exp_matrix.A, exp_rows, exp_cols)


def test_load_with_column_filter():
    # test if constraining the vocabulary a bit correctly drops columns
    t = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0c.strings',
                           column_filter=lambda x: x in {'a/N', 'b/V', 'd/J', 'g/N'})
    expected_matrix = np.array([
        [0.1, 0., 0.2, 0.8],  # ab
        [0., 0.1, 0.5, 0.3],  # a
        [0.1, 0., 0.3, 0.6],  # b
        [0.5, 0.3, 0., 0.7],  # d
        [0.3, 0.6, 0.7, 0.]  # g
    ])
    mat, cols, rows = t.to_sparse_matrix()
    assert set(cols) == {'a/N', 'b/V', 'd/J', 'g/N'}
    assert mat.shape == (5, 4)
    np.testing.assert_array_almost_equal(expected_matrix.sum(axis=0), mat.A.sum(axis=0))

    # test if severely constraining the vocabulary a bit correctly drops columns AND rows
    t = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0c.strings',
                           column_filter=lambda x: x in {'x/X'})
    mat, cols, rows = t.to_sparse_matrix()
    assert set(cols) == {'x/X'}
    assert mat.A == np.array([0.9])


def test_load_with_row_filter():
    # test if constraining the vocabulary a bit correctly drops columns
    t = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0c.strings',
                           row_filter=lambda x, y: x in {'a/N', 'd/J', 'g/N'})
    expected_matrix = np.array([
        [0., 0.1, 0.5, 0.3, 0.],  # a
        [0.5, 0.3, 0., 0.7, 0.],  # d
        [0.3, 0.6, 0.7, 0., 0.9]  # g
    ])
    mat, cols, rows = t.to_sparse_matrix()
    assert set(cols) == {'a/N', 'b/V', 'd/J', 'g/N', 'x/X'}
    assert set(rows) == {'a/N', 'd/J', 'g/N'}
    np.testing.assert_array_almost_equal(expected_matrix.sum(axis=0), mat.A.sum(axis=0))


def test_load_with_max_num_neighbours():
    t = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0c.strings',
                           max_neighbours=1)
    assert all(len(neigh) == 1 for neigh in t.values())
    mat, cols, rows = t.to_sparse_matrix()
    assert set(rows) == set(['g/N', 'a/N', 'd/J', 'b/V', 'a/J_b/N'])
    assert cols == ['d/J', 'g/N', 'x/X']


def test_max_num_neighbours_and_no_lexical_overlap():
    # max_neighbours filtering should kick in after lexical overlap filtering
    t = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0d.strings',
                           allow_lexical_overlap=False)
    assert len(t) == 4
    assert len(t['trade/N_law/N']) == 1
    assert len(t['prince/N_aziz/N']) == 3
    assert len(t['important/J_country/N']) == 5
    assert len(t['foreign/J_line/N']) == 3

    t = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0d.strings',
                           allow_lexical_overlap=False,
                           max_neighbours=1)
    assert len(t) == 4
    assert len(t['trade/N_law/N']) == 1
    assert t['trade/N_law/N'][0][0] == 'product/N_line/N'
    assert len(t['prince/N_aziz/N']) == 1
    assert len(t['important/J_country/N']) == 1
    assert len(t['foreign/J_line/N']) == 1

    t = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0d.strings')
    assert t['trade/N_law/N'][0][0] == 'law/N'
    assert t['trade/N_law/N'][4][0] == 'product/N_line/N'

    t = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0d.strings',
                           allow_lexical_overlap=True,
                           max_neighbours=1)
    assert t['trade/N_law/N'][0][0] == 'law/N'
    assert len(t['trade/N_law/N']) == 1


def test_loading_from_gz():
    t1 = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0a.strings')
    t2 = Thesaurus.from_tsv('discoutils/tests/resources/exp0-0a.strings.gzip')
    for k, v in t1.items():
        assert k in t2
        assert v == t2[k]


def test_loading_from_h5():
    t1 = Vectors.from_tsv('discoutils/tests/resources/exp0-0a.strings')
    t2 = Vectors.from_tsv('discoutils/tests/resources/exp0-0a.strings.h5')
    for k in t1.keys():
        assert k in t2
        v1 = t1.get_vector(k)
        v2 = t2.get_vector(k)
        np.testing.assert_array_equal(v1.A, v2.A)


def test_from_pandas_data_frame(vectors_c):
    mat, cols, rows = vectors_c.to_sparse_matrix()
    df = DataFrame(mat.A, index=rows, columns=cols)
    v = Vectors.from_pandas_df(df)

    mat1, cols1, rows1 = vectors_c.to_sparse_matrix()
    assert rows == rows1
    assert cols == cols1
    np.testing.assert_almost_equal(mat.A, mat1.A)

    vectors_c.init_sims()
    v.init_sims()
    for entry in vectors_c.keys():
        np.testing.assert_almost_equal(vectors_c.get_vector(entry).A,
                                       v.get_vector(entry).A)

        n1 = [x[0] for x in vectors_c.get_nearest_neighbours(entry)]
        n2 = [x[0] for x in v.get_nearest_neighbours(entry)]
        print(entry, n1, n2)
        assert n1 == n2


class TestLoad_thesauri(TestCase):
    def setUp(self):
        """
        Sets the default parameters of the tokenizer and reads a sample file
        for processing
        """

        self.tsv_file = 'discoutils/tests/resources/exp0-0a.strings'
        self.params = {'sim_threshold': 0, 'include_self': False}
        self.thesaurus = Thesaurus.from_tsv(self.tsv_file, **self.params)

    def _reload_thesaurus(self):
        self.thesaurus = Thesaurus.from_tsv(self.tsv_file, **self.params)

    def _reload_and_assert(self, entry_count, neighbour_count):
        th = Thesaurus.from_tsv(self.tsv_file, **self.params)
        all_neigh = [x for v in th.values() for x in v]
        self.assertEqual(len(th), entry_count)
        self.assertEqual(len(all_neigh), neighbour_count)
        return th

    def test_from_dict(self):
        from_dict = Thesaurus(self.thesaurus._obj)
        self.assertDictEqual(self.thesaurus._obj, from_dict._obj)

        # should raise KeyError for unknown tokens
        with self.assertRaises(KeyError):
            self.thesaurus['kasdjhfka']

    def test_from_shelved_dict(self):
        filename = 'thesaurus_unit_tests.tmp'
        self.thesaurus.to_shelf(filename)

        d = shelve.open(filename, flag='r')  # read only
        from_shelf = Thesaurus(d)
        for k, v in self.thesaurus.items():
            self.assertEqual(self.thesaurus[k], from_shelf[k])

        def modify():
            from_shelf['some_value'] = ('should not be possible', 0)

        self.assertRaises(Exception, modify)

        # tear down

        # on some systems a suffix may be added to the shelf file or the DB may be split in several files
        self.assertTrue(len(glob('%s*' % filename)))
        d.close()
        if os.path.exists(filename):
            os.unlink(filename)

    def test_sim_threshold(self):
        for i, j, k in zip([0, .39, .5, 1], [7, 3, 3, 0], [14, 4, 3, 0]):
            self.params['sim_threshold'] = i
            self._reload_thesaurus()
            self._reload_and_assert(j, k)

    def test_include_self(self):
        for i, j, k in zip([False, True], [7, 7], [14, 21]):
            self.params['include_self'] = i
            self._reload_thesaurus()
            th = self._reload_and_assert(j, k)

            for entry, neighbours in th.items():
                self.assertIsInstance(entry, str)
                self.assertIsInstance(neighbours, list)
                self.assertIsInstance(neighbours[0], tuple)
                if i:
                    self.assertEqual(entry, neighbours[0][0])
                    self.assertEqual(1, neighbours[0][1])
                else:
                    self.assertNotEqual(entry, neighbours[0][0])
                    self.assertGreaterEqual(1, neighbours[0][1])

    def test_iterate_nonoverlapping_pairs(self):
        inp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        output1 = [x for x in walk_nonoverlapping_pairs(inp, 1)]
        self.assertListEqual([(1, 2), (3, 4), (5, 6), (7, 8)], output1)

        output1 = [x for x in walk_nonoverlapping_pairs(inp, 1, max_pairs=2)]
        self.assertListEqual([(1, 2), (3, 4)], output1)

        output1 = [x for x in walk_nonoverlapping_pairs(inp, 1, max_pairs=-2)]
        self.assertListEqual([], output1)
