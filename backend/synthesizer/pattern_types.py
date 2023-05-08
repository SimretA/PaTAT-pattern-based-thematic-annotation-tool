#TYPE DEFINITIONS
POS = "POS"
WILD = "WILDCARD"
LITERAL = "LITERAL"
OPTIONAL = "OPTIONAL"
ENTITY = "ENTITY"

class stru:
    def __init__(self, type_, value_1, value_2=None):
        self.type_ = type_
        self.value_1 = value_1
        self.value_2 = value_2

class Pattern:
    def __init__(self, pattern, has_softmatch=False, softmatch_literal=[]):
        self.pattern = pattern
        self.has_softmatch = has_softmatch
        self.softmatch_literal
    
    def pat_to_object(self, pattern):
        pass
