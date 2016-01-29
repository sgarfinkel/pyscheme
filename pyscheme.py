from sys import argv
import operator
import math
import readline

# This will go in a separate module eventually
# Along with the dictionary of classes
# All scheme functions expect parameters as a list
# And parse each one accordingly
def scheme_display(params):
    print params[0].lstrip('"').rstrip('"')
    return None

# Basic math
def scheme_add(params):
    return sum(params)

def scheme_subtract(params):
    return params[0] - sum(params[1:])

def scheme_mult(params):
    return reduce(lambda x, y: x*y, params)

def scheme_div(params):
    v = reduce(lambda x, y: float(x)/y, params)
    if v.is_integer():
        return int(v)
    return v

# Comparators
def scheme_eq(params):
    return len(set(params)) == 1

def scheme_lt(params):
    return all([x[i] < x[i+1] for i, x in enumerate(params[1:])])

def scheme_gt(params):
    return all([x[i] > x[i+1] for i, x in enumerate(params[1:])])

def scheme_lte(params):
    return not scheme_gt(params)

def scheme_gte(params):
    return not scheme_lt(params)

# Logical operators
def scheme_and(params):
    return all(params)

def scheme_or(params):
    return any(params)

def scheme_not(params):
    return not params[0]

# Environment
class Env:
    def __init__(self):
        self.env = dict()
        self.parent = None

    def __setitem__(self, key, val):
        self.env[key] = val

    # Traverse the linked list to find the outermost environment
    # Containing the binding we are looking for
    def find(self, name):
        if not name in self.env:
            return self.parent.find(name)
        return self.env[name]

def std_env():
    env = Env()
    env.env = { 'display'   : scheme_display,
                '+'         : scheme_add,
                '-'         : scheme_subtract,
                '*'         : scheme_mult,
                '/'         : scheme_div,
                '='         : scheme_eq,
                '<'         : scheme_lt,
                '>'         : scheme_gt,
                '<='        : scheme_lte,
                '>='        : scheme_gte,
                'and'       : all,
                'or'        : any,
                'not'       : scheme_not,
            }
    env.env.update(vars(math))
    return env

global_env = std_env()

# Binding
class Binding:
    def __init__(self, name=None, expr=None, env=None):
        self.name = name[0]
        self.params = name[1:]
        self.expr = expr
        self.env = env

    def __repr__(self):
        return 'name='+str(self.name)+';params='+str(self.params)+';expr='+str(self.expr)

    # Private eval method, only visible if the binding is a function
    def _eval(self, params=None):
        for k, v in zip(self.params, params):
            self.env[k] = v
        return eval_expr(self.expr, self.env)

    # Helper method--we don't hold onto the reference to the instance of the binding
    # Only the method to evaluate, so we can't do type checking of the expression
    # Instead, we use duck typing here to add the binding to the environment as a literal
    # Or as an Python function
    def eval(self):
        if self.params:
            return self._eval
        else:
            return self.expr


# Parser
def tokenize(text):
    tokens = list()
    cur_token = list()
    in_string = False
    q_count = 0
    quote = False
    for char in text:
        # Handle whitespace
        if (char == ' ' or char == '\n') and not in_string:
            # Only append if we have symbols to add
            # Catches instances of leading whitespace
            if len(cur_token):
                tokens.append(''.join(cur_token))
                # And clear the cur_token
                cur_token = list()
            # Regardless, continue
            continue
        # Handle parentheses, special case that may not have whitespace
        # Other than white space, only other time to add new token
        if char == '(' or char == ')':
            # If we already have tokens to add, add them first
            if len(cur_token):
                tokens.append(''.join(cur_token))
                # And clear the cur_token
                cur_token = list()
            tokens.append(char)
            # Handle quote symbol, special case
            if char == '(' and quote:
                q_count += 1
            elif char == ')' and quote:
                q_count -= 1
            if q_count == 0 and quote:
                quote = not quote
                tokens.append(')')
            continue
        # To make parsing easier, convert quote symbols
        # Into their functional form
        if char == '\'' or char == '`':
            tokens.extend(['(', 'quote'])
            quote = True
            continue
        # All other cases, append char
        cur_token.append(char)
        # A quotation mark starts or ends a string
        # If it ends a string, just append and start a new token
        # May help with catching errors down the road
        if char == '"':
            if in_string:
                tokens.append(''.join(cur_token))
                cur_token = list()
            in_string = not in_string

    # When we're done, if we have a cur_token, append
    # This is mainly for repl if you call a literal on its own
    if cur_token:
        tokens.append(''.join(cur_token))

    # Return our tokens
    return tokens

# This recursively parses our tokens
def parse(tokens):
    token = tokens.pop(0)
    if token == '(':
        parsed = []
        while tokens[0] != ')':
            parsed.append(parse(tokens))
        tokens.pop(0)
        return parsed
    else:
        return cast(token)

# Typecasting of tokens
def cast(token):
    try:
        v = float(token)
        if v.is_integer():
            return int(v)
    except ValueError:
        # Bools
        if token == '#t':
            return True
        elif token == '#f':
            return False
        # Everything else, including string literals
        # Functions that work with strings can check for encapsulation in quotes
        else:
            return token

def to_scheme(val):
	# If input is literal True
	if val is True:
		return '#t'
	# If input is literal False
	elif val is False:
		return '#f'
	else:
		return val

def eval_expr(elem, env=global_env):
    local_env = Env()
    local_env.parent = env
    print elem
    # Any element that is a string should be in the environment
    if isinstance(elem, str) and not (elem.startswith('"') and elem.endswith('"')):
        return local_env.find(elem)
    # Literals are anything that isn't a list or string
    elif not isinstance(elem, list):
        return elem
    # Conditionals
    elif elem[0] == 'if': # (if test conseq alt)
        (func, test, conseq, alt) = elem
        ret = (conseq if eval_expr(test, local_env) else alt)
        return eval_expr(ret, local_env)
    # Any list is a function, unless it's a literal list
    elif elem[0] == 'quote': # '(v1...) or (quote (v1...))
        return elem[1:]
    # Define
    elif elem[0] == 'define': #(define name expr) or (define (name params) expr)
        (func, name, expr) = elem
        bind = Binding(name, expr, local_env)
        global_env[bind.name] = bind.eval()
    # Lambda
    elif elem[0] == 'lambda': #(lambda (params) (expr))
        (func, name, expr) = elem
        name.insert(0, None) # A lambda function has no name, only params
        bind = Binding(name, expr, local_env)
        return bind.eval()
    # Otherwise it's in env, and we can evaluate it
    else:
        func = eval_expr(elem[0], local_env)
        body = [eval_expr(e, local_env) for e in elem[1:]]
        return func(body)


# Read, eval, print
def rep(text):
    tokens = tokenize(text)
    expr = parse(tokens)
    ret = eval_expr(expr)
    if ret != None:
        print to_scheme(ret)

# Read, eval, print loop
def repl():
    while True:
        text = raw_input('pyscheme>')
        if text == '(exit)':
            break
        rep(text)

# Main line
if __name__ == '__main__':
    if len(argv) == 2:
        # Open as a file
        with open(argv[1]) as f:
            rep(f.read())
    else:
        repl()