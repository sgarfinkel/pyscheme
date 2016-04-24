import argparse
from sys import stdout, stdin
import operator
import math
import readline
from socket import socket, AF_INET, SOCK_STREAM
import threading
from time import sleep

# I/O
def scheme_display(s, f=stdout, *args):
    f.write(str(s).strip('"')+'\n')
    return None

def scheme_open_output_file(fname):
    return open(fname.strip('"'), 'w')

def scheme_close_output_port(f):
    close(f)
    return None

def scheme_open_input_file(fname):
    return open(fname.strip('"'), 'r')

def scheme_close_input_port(f):
    close(f)
    return None

def scheme_read_line(f):
    return '"' + f.readline() + '"'

def scheme_read(f):
    return '"' + f.read() + '"'

# Basic math
def scheme_add(*args):
    return sum(args)

def scheme_subtract(*args):
    return args[0] - sum(args[1:])

def scheme_mult(*args):
    return reduce(lambda x, y: x*y, args)

def scheme_div(*args):
    v = reduce(lambda x, y: float(x)/y, args)
    if v.is_integer():
        return int(v)
    return v

# Comparators
def scheme_eq(*args):
    return len(set(args)) == 1

def scheme_lt(*args):
    return all([args[i] < args[i+1] for i, x in enumerate(args[:1])])

def scheme_gt(*args):
    return all([args[i] > args[i+1] for i, x in enumerate(args[:1])])

def scheme_lte(*args):
    return not scheme_gt(args)

def scheme_gte(*args):
    return not scheme_lt(args)

# Logical operators
def scheme_and(*args):
    return all(args)

def scheme_or(*args):
    return any(args)

def scheme_not(arg):
    return not arg

# Sockets
def scheme_socket_create(host, port):
    '''Create a socket object. Only supports IPV4.'''
    host = host.strip('"')
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind((host, port))
    return sock

def scheme_socket_close(sock):
    '''Closes a socket object.'''
    sock.close()
    return None

def scheme_socket_accept(sock):
    '''Listens for a socket connection. Returns a socket object for the connection after accepting.'''
    sock.listen(1)
    (conn, addr) = sock.accept()
    return conn

def scheme_socket_readline(conn):
    '''Listens for a newline terminated string on the connection object and returns it. Strips the terminating newline.'''
    chunks = list()
    while True:
        chunk = conn.recv(4096)
        chunks.append(chunk)
        if chunk.endswith('\n'):
            break
    return ''.join(chunks).rstrip('\n')

def scheme_socket_read(conn, num_bytes):
    '''Reads b bytes from socket object conn. Returns the bytes as a string.'''
    chunks = list()
    while True:
        chunks.append(conn.recv(1))
        if len(chunks) == num_bytes:
            break
    return ''.join(chunks)

def scheme_socket_write(conn, s):
    '''Writes argument s to the conn socket object.'''
    conn.sendall(string.strip('"'))
    return None

# String manipulation
def scheme_string_to_number(s):
    return int(s)

def scheme_number_to_string(n):
    return str(n)

def scheme_string_to_list(s):
    return list(s)

def scheme_list_to_string(l):
    return ''.join(s)

def scheme_substring(s, start, end):
    return s[start:end]

def scheme_string_length(s):
    return len(s)

def scheme_string_append(*args):
    return ''.join(args)

def scheme_string_join(strs, sep):
    '''Joins each string in the strs list together with the separator.
    Returns a string consisting of the concatenation of the new list.'''
    return sep.join(strs)

# List manipulation
def scheme_list(*args):
    return args

def scheme_length(l):
    return len(l)

def scheme_car(l):
    '''Returns the first element of the list. Error if l is empty.'''
    return l[0]

def scheme_cdr(l):
    '''Returns the sublist obtained by removing the first element.'''
    return l[1:]

def scheme_append(*args):
    l = list()
    for a in args:
        if isinstance(a, list):
            l.extend(a)
        else:
            l.append(a)
    return l

# List iteration operators
def scheme_map(func, vals):
    '''Applies function func to each element of vals and returns a new list consisting of the
    output of each execution of procedure proc.'''
    return [func(v) for v in vals]

def scheme_for_each(func, vals):
    '''Applies function func to each elementent of val. The return values of func are ignored.'''
    for v in vals:
        func(v)
    return None

# Threads
def scheme_thread_create(expr):
    '''Creates a new thread (but does not execute it)
    with the given expression. Returns the thread object.'''
    t = threading.Thread(target=expr)
    t.start()
    return t

def scheme_thread_run(t):
    '''Executes the function associated with the thread object t.'''
    t.run()
    return None

def scheme_thread_join(t):
    '''Blocks the caller until the callee thread terminates.'''
    t.join()
    return None

def scheme_thread_sleep(dur):
    '''Sleeps this thread for the given duration dur.'''
    sleep(dur)
    return None

def scheme_lock_create():
    return threading.Lock()

def scheme_lock_acquire(lock):
    lock.acquire()
    return None

def scheme_lock_release(lock):
    lock.release()
    return None

# Environment
class Env:
    def __init__(self):
        self.env = dict()
        self.parent = None

    def __setitem__(self, key, val):
        self.env[key] = val

    def update(self, d):
        self.env.update(d)

    # Traverse the linked list to find the outermost environment
    # Containing the binding we are looking for
    def find(self, name):
        if not name in self.env:
            if self.parent:
                return self.parent.find(name)
            print 'Unbound function or variable: {0}'.format(name)
            return None
        return self.env[name]

def std_env():
    env = Env()
    env.env = { 'void'              : lambda *args: None,
                'display'           : scheme_display,
                'open-output-file'  : scheme_open_output_file,
                'close-output-port' : scheme_close_output_port,
                'open-input-file'   : scheme_open_input_file,
                'close-input-port'  : scheme_close_input_port,
                'read-line'         : scheme_read_line,
                'read'              : scheme_read,
                '+'                 : scheme_add,
                '-'                 : scheme_subtract,
                '*'                 : scheme_mult,
                '/'                 : scheme_div,
                '='                 : scheme_eq,
                '<'                 : scheme_lt,
                '>'                 : scheme_gt,
                '<='                : scheme_lte,
                '>='                : scheme_gte,
                'and'               : scheme_and,
                'or'                : scheme_or,
                'not'               : scheme_not,
                'socket-create'     : scheme_socket_create,
                'socket-close'      : scheme_socket_close,
                'socket-accept'     : scheme_socket_accept,
                'socket-readline'   : scheme_socket_readline,
                'socket-read'       : scheme_socket_read,
                'socket-write'      : scheme_socket_write,
                'string->number'    : scheme_string_to_number,
                'number->string'    : scheme_number_to_string,
                'string->list'      : scheme_string_to_list,
                'list->string'      : scheme_list_to_string,
                'substring'         : scheme_substring,
                'string-length'     : scheme_string_length,
                'string-append'     : scheme_string_append,
                'string-join'       : scheme_string_join,
                'list'              : scheme_list,
                'length'            : scheme_length,
                'car'               : scheme_car,
                'cdr'               : scheme_cdr,
                'append'            : scheme_append,
                'map'               : scheme_map,
                'for-each'          : scheme_for_each,
                'thread-create'     : scheme_thread_create,
                'thread-run'        : scheme_thread_run,
                'thread-join'       : scheme_thread_join,
                'thread-sleep'      : scheme_thread_sleep,
                'lock-create'       : scheme_lock_create,
                'lock-acquire'      : scheme_lock_acquire,
                'lock-release'      : scheme_lock_release,
            }
    env.update(vars(math))
    return env

global_env = std_env()

# Binding
class Binding:
    def __init__(self, name=None, expr=None, env=None):
        if isinstance(name, list):
            self.name = name[0]
            self.params = name[1:]
        else:
            self.name = name
            self.params = None
        self.expr = expr
        self.env = env

    def __repr__(self):
        return 'name='+str(self.name)+';params='+str(self.params)+';expr='+str(self.expr)

    def _eval(self, *args):
        for k, v in zip(self.params, args):
            self.env[k] = v
        return eval_expr(self.expr, self.env)

    def eval(self):
        if self.params:
            return self._eval
        else:
            return eval_expr(self.expr, self.env)


def tokenize(text):
    tokens = list()
    token = list()
    in_str = False
    quote = False
    q_count = 0
    for char in text:
        if char == '"':
            token.append(char)
            if in_str:
                tokens.append(''.join(token))
                token = list()
            in_str = not in_str
        elif in_str:
            token.append(char)
        elif char == '(' or char == ')':
            if len(token):
                tokens.append(''.join(token))
                token = list()
            tokens.append(char)
            if char == '(' and quote:
                q_count += 1
            elif char == ')' and quote:
                q_count -= 1
            if q_count == 0 and quote:
                quote = not quote
                tokens.append(')')
        elif char == '\'' or char == '`':
            quote = not quote
            tokens.extend(['(', 'quote'])
        elif not char in [' ', '\n', '\r', '\t']:
            token.append(char)
        else:
            if len(token):
                tokens.append(''.join(token))
                token = list()
    if token:
        tokens.append(''.join(token))
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
        return v
    except ValueError:
        # Bools
        if token == '#t':
            return True
        elif token == '#f':
            return False
        # Everything else, including string literals
        # Functions that work with strings can check for encapsulation in quotes
        else:
            # Handle string literals
            if token.startswith('"') and token.endswith('"'):
                token = token.replace('\\n', '\n')
                token = token.replace('\\r', '\r')
                token = token.replace('\\t', '\t')
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
        return elem[1]
    # Define
    elif elem[0] == 'define': # (define name expr) or (define (name params) expr)
        (func, name, expr) = elem
        bind = Binding(name, expr, local_env)
        global_env[bind.name] = bind.eval()
        return None
    # Lambda
    elif elem[0] == 'lambda': # (lambda (params) (expr))
        (func, name, expr) = elem
        name.insert(0, None) # A lambda function has no name, only params
        bind = Binding(name, expr, local_env)
        return bind.eval()
    # Let
    elif elem[0] == 'let': # (let ((a expr_a) (b expr_b)) bodys...) or (let func (id expr...) (bodys...))
        func_name = list()
        if isinstance(elem[1], list):
            bindings = elem[1]
            bodys = elem[2:]
        else:
            func_name.append(elem[1])
            bindings = elem[2]
            bodys = elem[3:]
        # In let, we first create each binding, then we add to the local_env
        # The bindings are not added as they are made
        # A binding for let consists of a list of length 2
        local_binds = dict()
        for b in bindings:
            (b_name, b_expr) = b
            # If we are binding a new function, we need to store the names of the bindings to act as parameters
            if func_name:
                func_name.append(b_name)
            local_binds[b_name] = Binding(b_name, b_expr, local_env).eval()
        local_env.update(local_binds)
        # Now bind the function if necessary to the list of bodys
        # Expression is of type begin, as that supports execution of a list of expressions
        if func_name:
            expr = ['begin']
            [expr.append(e) for e in bodys]
            b = Binding(func_name, expr, local_env)
            local_env[b.name] = b.eval()
        # Return from the last expression in the set of bodys
        return [eval_expr(e, local_env) for e in bodys][-1]
    # Let*
    # Bindings are immediately added to the current local environment
    # As they are created. This makes them available to subsequent bindings
    elif elem[0] == 'let*':
        (func, bindings, expr) = elem
        for b in bindings:
            (b_name, b_expr) = b
            local_env[b_name] = Binding(b_name, b_expr, local_env).eval()
        return eval_expr(expr, local_env)
    # Begin
    # Sequentially executes each function
    # Returns the output of the last function
    elif elem[0] == 'begin':
        out = None
        for expr in elem[1:]:
            out = eval_expr(expr, local_env)
        return out
    # Otherwise it's in env, and we can evaluate it
    else:
        func = eval_expr(elem[0], local_env)
        body = [eval_expr(e, local_env) for e in elem[1:]]
        return func(*body)

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
        else:
            # We don't implement our own error handling
            # But we don't want exceptions to kill the repl
            try:
                rep(text)
            except Exception:
                pass

# Main line
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store_true', dest='import_env',
        help='executes the file from the -f flag then starts an interactive shell with the old environment')
    parser.add_argument('-f', default=None, help='executes the given scheme file', metavar='filename')
    args = parser.parse_args()
    if args.f:
        with open(args.f) as f:
            rep(f.read())
        if args.import_env:
            repl()
    else:
        repl()
