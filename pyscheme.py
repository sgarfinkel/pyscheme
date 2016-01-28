from sys import argv

# This will go in a separate module eventually
# Along with the dictionary of classes
# All scheme functions expect parameters as a list
# And parse each one accordingly
def scheme_display(params):
    if len(params) > 1:
        raise(Exception)
    print params[0]
    return None

def scheme_quote(params):
    return params

# Basic Math
def scheme_add(params):
    return sum(params)

def scheme_subtract(params):
    return params[0] - sum(params[1:])

def scheme_mult(params):
    return reduce(lambda x, y: x*y, params)

def scheme_div(params):
    return reduce(lambda x, y: float(x)/y, params)

func_dict = {'display'  : scheme_display,
             'quote'    : scheme_quote,
             '+'        : scheme_add,
             '-'        : scheme_subtract,
             '*'        : scheme_mult,
             '/'        : scheme_div, 
             }

# FOR DEBUGGING
# Helper, not sure where to put this
def pretty(l):
    string = []
    for e in l:
        if isinstance(e, list):
            string.append(pretty(e))
        else:
            string.append(str(e))
    return ''.join(['\'(', ' '.join(string), ')'])

# Not sure if we need this
# It's nice to have a defined type for functions, though
class SchemeFunction:
    def __init__(self):
        pass

# A generic symbol class, can be a primitive or instance of type SchemeFunction
class Symbol:
    def __init__(self, val, type=None):
        self.val = val
        self.name = None
        self.params = None
        self.type = type
        # Type cast
        # Just support strings, integers, and floats for now
        if not type:
            try:
                self.val = float(self.val)
                self.type = float
                if self.val.is_integer():
                    self.val = int(self.val)
                    self.type = int
            except ValueError:
                if val.startswith('"') and val.endswith('"'):
                    self.val = val.lstrip('"').rstrip('"')
                    self.type = str
                # Else it's type SchemeFunction
                else:
                    self.val = func_dict[val]
                    self.type = SchemeFunction

    def __repr__(self):
        if self.type == SchemeFunction:
            return '<'+self.val.__name__+'>'
        return '<'+str(self.val)+'>'

    def __str__(self):
        return str(self.val)

    def __call__(self):
        if self.type == SchemeFunction:
            return self.val(self.params)
        return self.val

    def set_params(self, params):
        self.params = params

    def type(self):
        return self.type

def tokenize(text):
    tokens = list()
    cur_token = list()
    in_string = False
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
            continue
        # Handle quote symbol, special case
        if char == '\'' or char == '`':
            tokens.append(char)
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

    # Return our tokens
    return tokens

def _parse(tokens, i):
    parsed = list()
    while i < len(tokens):
        t = tokens[i]
        # An open parenthesis denotes a new list
        if t == '(':
            (res, k) = _parse(tokens, i+1)
            # If we have an empty list already, just replace
            if not parsed:
                parsed = res
            # Otherwise, this needs to be appended
            else:
                parsed.append(res)
            i = k
        # If we get a close parenthesis we should return parsed
        elif t == ')':
            break
        # Turn single quotes into a quote function
        elif t == '\'' or t == '`':
            # First parse the remaining tokens
            (res, k) = _parse(tokens, i+1)
            quote = [Symbol('quote'), res]
            if not parsed:
                parsed = quote
            else:
                parsed.append(quote)
            i = k
        else:
            parsed.append(Symbol(t))
        i += 1

    return (parsed, i)


# This recursively parses our tokens
def parse(tokens):
    (out, k) = _parse(tokens, 0)
    return out

# Evaluate the expression
def eval_expr(expr):
    env = list()
    func = None
    for t in expr:
        if hasattr(t, 'type'):
            if t.type == SchemeFunction:
                func = t.val
            else:
                env.append(t.val)
        else:
            if func == scheme_quote:
                env = t
            else:
                env.append(eval_expr(t))
    return func(env)



# Read, eval, print
def rep(text):
    tokens = tokenize(text)
    expr = parse(tokens)
    ret = eval_expr(expr)
    if ret:
        print ret

# Read, eval, print loop
def repl():
    while True:
        text = raw_input()
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