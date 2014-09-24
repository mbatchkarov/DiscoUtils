from collections import Counter
from pprint import pprint

__author__ = 'miroslavbatchkarov'
from discoutils.find_all_NPs import get_NPs, get_window_vectors

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


def test_find_all_NPs():
    s = StringIO()
    get_NPs('discoutils/tests/resources/exp10head.pbfiltered', s)
    expected = "full-time/J_tribunal/N\n" \
               "ordinary/J_session/N\n" \
               "council/N_session/N\n" \
               "large/J_cat/N\n" \
               "fluffy/J_cat/N\n" \
               "house/N_cat/N\n" \
               "street/N_cat/N\n" \
               "troubled/J_activist/N\n" \
               "leftw/J_activist/N\n"
    assert s.getvalue() == expected


def test_find_all_NPs_with_seed():
    s = StringIO()
    get_NPs('discoutils/tests/resources/exp10head.pbfiltered', s, whitelist={'ordinary/J'})
    expected = "ordinary/J_session/N\n"
    assert s.getvalue() == expected


def test_find_all_NPs_with_window_vectors():
    s = StringIO()
    get_window_vectors('discoutils/tests/resources/exp10head.pbfiltered', s)
    expected_entries = Counter({'council/N_session/N': 2,
                                "large/J_cat/N": 1,
                                "fluffy/J_cat/N": 1,
                                "house/N_cat/N": 1,
                                "street/N_cat/N": 1,
                                "troubled/J_activist/N": 1,
                                "leftw/J_activist/N": 1,
                                "ordinary/J_session/N": 1
    })

    expected_features = {'T:something/N', 'T:large/J', 'T:what/CONJ', 'T:ever/N', 'T:feature/J', 'T:cat/N'}
    lines = s.getvalue().rstrip().split('\n')
    pprint(lines)
    assert len(lines) == 9
    # only some NPs have window features. Some lines in the test
    # file mention multiple NPs- all need to be produced. Some NNs appear twice.
    # Just check that the right features are returned, by do not bother with checking their ordering
    assert expected_features == set(feature for line in lines for feature in line.split('\t')[1:])
    assert expected_entries == Counter(line.split('\t')[0] for line in lines)


def test_find_all_NPs_with_window_vectors_with_filtering():
    s = StringIO()
    get_window_vectors('discoutils/tests/resources/exp10head.pbfiltered', s,
                   whitelist={'council/N_session/N', "large/J_cat/N"})
    expected_entries = Counter({'council/N_session/N': 2,
                                "large/J_cat/N": 1})

    expected_features = {'T:something/N', 'T:large/J', 'T:cat/N'}
    lines = s.getvalue().rstrip().split('\n')
    pprint(lines)
    assert len(lines) == 3
    assert expected_features == set(feature for line in lines for feature in line.split('\t')[1:])
    assert expected_entries == Counter(line.split('\t')[0] for line in lines)