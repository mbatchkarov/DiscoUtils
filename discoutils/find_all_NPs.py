import argparse
import re
import logging

from misc import ContainsEverything

'''
Finds all adjective-noun and noun-noun compounds in a FET output file.
Optionally accepts only these ANs/NNs whose modifier is mentioned in a file.
Requires features to have an associated PoS tag, see
discoutils/tests/resources/exp10head.pbfiltered
'''

noun_pattern = re.compile('^(\S+?/N).+')  # a noun entry
adj_pattern = re.compile('^(\S+?/J).+')  # an adjective entry

# an adjective modifier in an amod relation, consisting of non-whitespace
an_modifier_feature_pattern = re.compile('.*?amod-DEP:(\S+/J).*?')
an_head_feature_pattern = re.compile('.*?amod-HEAD:(\S+/N).*?')  # an adjective modifier/head in an amod relation

nn_modifier_feature_pattern = re.compile('.*?nn-DEP:(\S+/N).*?')  # an noun modifier in a nn relation
nn_head_feature_pattern = re.compile('.*?nn-HEAD:(\S+/N).*?')  # an noun head in a nn relation

window_feature_pattern = re.compile('.*?(T:\S+.*?)')  # an noun in a nn relation


def _find_and_output_features(head, line, modifier, np_type, outstream, modifier_set):
    if np_type and head and modifier:
        # if we've found en entry having the right PoS and dep features, get its window features
        features = window_feature_pattern.findall(line)
        if features and modifier in modifier_set:
            outstream.write('{}:{}_{}\t{}\n'.format(np_type, modifier, head, '\t'.join(features)))


def go_get_vectors(infile, outstream, seed_set=ContainsEverything()):
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % 100000 == 0:
                logging.info('Done %d lines', i)

            np_type, head, modifier = None, None, None
            # check if the entry is an adj or a noun
            noun_match = noun_pattern.match(line)
            adj_match = adj_pattern.match(line)
            amod_head_match = an_head_feature_pattern.match(line)
            amod_modifier_match = an_modifier_feature_pattern.match(line)
            nn_head_match = nn_head_feature_pattern.match(line)
            nn_modifier_match = nn_modifier_feature_pattern.match(line)

            if adj_match and amod_modifier_match:
                logging.warn('Adjective has adjectival modifiers')
                np_type = None
                continue

            if adj_match and amod_head_match:
                modifier = adj_match.groups()[0]
                head = amod_head_match.groups()[0]
                np_type = 'AN'
                print modifier, head
                _find_and_output_features(head, line, modifier, np_type, outstream, seed_set)

            if noun_match and amod_modifier_match:
                head = noun_match.groups()[0]
                np_type = 'AN'
                for modifier in an_modifier_feature_pattern.findall(line):
                    _find_and_output_features(head, line, modifier, np_type, outstream, seed_set)

            if noun_match and nn_head_match:
                np_type = 'NN'
                head = noun_match.groups()[0]
                modifier = nn_head_match.groups()[0]
                print head, modifier
                _find_and_output_features(head, line, modifier, np_type, outstream, seed_set)

            if noun_match and nn_modifier_match:
                np_type = 'NN'
                head = nn_modifier_match.groups()[0]
                for modifier in nn_modifier_feature_pattern.findall(line):
                    print head, modifier
                    _find_and_output_features(head, line, modifier, np_type, outstream, seed_set)


def go(infile, outstream, seed_set=ContainsEverything()):
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % 100000 == 0:
                logging.info('Done %d lines', i)

            noun_match = noun_pattern.match(line)
            if noun_match:
                head = noun_match.groups()[0]
                for np_type, pattern in zip(['AN', 'NN'],
                                            [an_modifier_feature_pattern, nn_modifier_feature_pattern]):
                    for modifier in pattern.findall(line):
                        if modifier in seed_set:
                            outstream.write('{}:{}_{}\n'.format(np_type, modifier, head))


def read_configuration():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Input file')
    parser.add_argument('-o', '--output', required=False, type=str, default=None,
                        help='Name of output file. Default <input file>.ANsNNs')
    parser.add_argument('-s', '--modifier_set', required=False, type=str, default='',
                        help='Name of file containing a set of modifiers that will be considered. '
                             'All other ANs/NNs are disregarded.')
    parser.add_argument('-v', '--vectors', action='store_true',
                        help='If set, will also output '
                             'window features for each entry occurrence .Default is False')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s """
                               "(line %(lineno)d)\t%(levelname)s : %(""message)s")
    conf = read_configuration()
    output = conf.output if conf.output else '%s.ANsNNs' % conf.input

    if conf.vectors:
        logging.info('Will also output window features for each entry occurrence')
        function = go_get_vectors
    else:
        logging.info('Will only output NP entries')
        function = go
    seed = conf.modifier_set

    if seed:
        with open(seed) as f:
            seed = set(x.strip() for x in f.readlines())

    with open(output, 'w') as outstream:
        logging.info('Reading from %s', conf.input)
        logging.info('Writing to %s', output)

        if seed:
            function(conf.input, outstream, seed_set=seed)
        else:
            function(conf.input, outstream)
