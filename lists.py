from sklearn.feature_extraction import text


##
##

# establishing the expanded_stopwords list

expanded_proper_names = ['seven', 'clone', 'warp', 'borg', 'trilogy', 'contact', 'anakin', 'paramount', 'leia',\
                         'kirk', 'wan', 'jedi', 'kenobi', 'snw', 'wars', 'vader', 'order', 'skywalker', 'klingon', 'starfleet',\
                         'ds9', 'captain', 'maul', 'luke', 'obi', 'rebels', 'data', 'voyager', 'st', 'discovery', 'federation',\
                         'pike', 'picard', 'mandalorian', 'klingons', 'star', 'tng', 'reva', 'strange', 'disney', 'worf',\
                         'riker', 'empire', 'jurati', 'palpatine', 'yoda', 'force', 'darth', 'republic', 'lightsaber', 'sith',\
                         'spock', 'boba', 'fett', 'inquisitor', 'trek', 'enterprise', 'tos', 'padme']

expanded_stopwords = text.ENGLISH_STOP_WORDS.union(expanded_proper_names)