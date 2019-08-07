class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        Your code should initialize the following fields:
            self.stack: The current stack represented as a list with the top of the stack as the
                        last element of the list.
            self.buffer: The current buffer represented as a list with the first item on the
                         buffer as the first item of the list
            self.dependencies: The list of dependencies produced so far. Represented as a list of
                    tuples where each tuple is of the form (head, dependent).
                    Order for this list doesn't matter.

        The root token should be represented with the string "ROOT"

        Args:
            sentence: The sentence to be parsed as a list of words.
                      Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes. Do not use it in your code.
        self.sentence = sentence

        ### YOUR CODE HERE
        self.stack = ['ROOT']
        self.buffer = []
        for w in sentence:
            self.buffer.append(w)
        self.dependencies = []
        ### END YOUR CODE

    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        Args:
            transition: A string that equals "S", "LA", or "RA" representing the shift, left-arc,
                        and right-arc transitions.
        """
        ### YOUR CODE HERE
        if transition == 'S':
            if len(self.buffer) >= 1:
                self.stack.append(self.buffer.pop(0))
        elif transition == 'LA':
            if len(self.stack) >= 2:  # what if a wrong transition is issued?
                self.dependencies.append((self.stack[-1], self.stack[-2]))
                self.stack.pop(-2)
        elif transition == 'RA':
            if len(self.stack) >= 2:
                self.dependencies.append((self.stack[-2], self.stack[-1]))
                self.stack.pop(-1)
        else:
            raise Exception('transition='+transition)
        ### END YOUR CODE

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        Args:
            transitions: The list of transitions in the order they should be applied
        Returns:
            dependencies: The list of dependencies produced when parsing the sentence. Represented
                          as a list of tuples where each tuple is of the form (head, dependent)
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies

import random
def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    Args:
        sentences: A list of sentences to be parsed (each sentence is a list of words)
        model: The model that makes parsing decisions. It is assumed to have a function
               model.predict(partial_parses) that takes in a list of PartialParses as input and
               returns a list of transitions predicted for each parse. That is, after calling
                   transitions = model.predict(partial_parses)
               transitions[i] will be the next transition to apply to partial_parses[i].
        batch_size: The number of PartialParses to include in each minibatch
    Returns:
        dependencies: A list where each element is the dependencies list for a parsed sentence.
                      Ordering should be the same as in sentences (i.e., dependencies[i] should
                      contain the parse for sentences[i]).
    """

    ### YOUR CODE HERE
    
    '''
    initialize partial_parses as a list of partial parses, one for each
    sentence in sentences
    '''
    partial_parses = []
    nn = []  # catch stalled
    kk = []
    for sen in sentences:
        partial_parses.append(PartialParse(sen))
        nn.append(len(sen))
        kk.append(0)

    '''
    initialize unfinished_parses as a shallow copy of partial_parses
    https://docs.python.org/2/library/copy.html
    but it turns partial_parses has no further use so this part is pointless
    '''
    #unfinished_parses = copy.copy(partial_parses)
    pps = partial_parses

    dependencies = [None]*len(sentences)
    old_indices = range(len(sentences))
    while len(pps) > 0:
        '''
        take the first batch_size parses in unfinished_parses as a minibatch
        note len(selected) < batch_size at the end
        '''
        if len(pps) >= batch_size:
            selected = random.sample(xrange(len(pps)), batch_size)
        else:
            selected = xrange(len(pps))

        '''
        use the model to predict the next transition for each partial parse in
        the minibatch
        '''
        trans = model.predict([pps[i] for i in selected])
        #print trans
        # after this, pps[0].stack[0] somehow became 0 (from 'ROOT')

        '''
        perform a parse step on each partial parse in the minibatch with its
        predicted transition
        '''
        for i in xrange(len(selected)):
            pps[selected[i]].parse_step(trans[i])
            kk[selected[i]] += 1  # catch stalled

        '''
        remove the completed parses from unfinished_parses
        '''
        for i in sorted(selected, reverse=True):
            pp = pps[i]
            if (len(pp.buffer) == 0 and len(pp.stack) == 1) or\
                kk[i] >= 2*nn[i]:  # catch stalled

                dependencies[old_indices[i]] = pp.dependencies
                pps.pop(i)
                old_indices.pop(i)
                #print float(kk[i])/float(nn[i])
                kk.pop(i)  # catch stalled
                nn.pop(i)

    '''
    moral:
    if there is pseudo code, type it down to follow it exactly
    
    use extreme caution when changing an array while looping through it
    
    the above is a good template for minibatch processing with async finish
    '''
    ### END YOUR CODE

    return dependencies


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print "{:} test passed!".format(name)


def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected,  \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print "parse test passed!"


class DummyModel:
    """Dummy model for testing the minibatch_parse function
    First shifts everything onto the stack and then does exclusively right arcs if the first word of
    the sentence is "right", "left" if otherwise.
    """
    def predict(self, partial_parses):
        return [("RA" if pp.stack[1] is "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]


def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))
    print "minibatch_parse test passed!"

if __name__ == '__main__':
    test_parse_step()
    test_parse()
    test_minibatch_parse()
