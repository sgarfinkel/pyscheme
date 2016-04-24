import argparse
from sys import stdout, stdin
import operator
import math
import readline
from socket import socket, AF_INET, SOCK_STREAM
import threading
from time import sleep
from traceback import print_exc
from functools import wraps
from pprint import pprint

# Decorator that sets an attribute to
# Distinguish macros from functions
def macro(func):
    @wraps(func)
    def wrapper(*args):
        return func(*args)
    wrapper.macro = True
    return wrapper

@macro
def scheme_define(args, env):
    """Define macro. Adds a new binding to the global environment."""
    first = args[0]
    # If the first argument is a list
    # Then we're binding a new function
    if isinstance(first, list):
        def inner(*params):
            for k,v in zip(first[1:], params):
                # Evaluate each input param
                # With the bound environment
                # Of the closure, and add these
                # Params to the closure's environment
                env[k] = v
            # Then this function should evaluate the body(s) and return the last
            return [eval_expr(a, env) for a in args[1:]][-1]
        # Now add this function bound with its name in the global_env
        env.root[first[0]] = inner
    # Else literal binding
    # Bind the name to the evaluation
    # Of the body--body is evaluated at creation
    # Given the environment of the closure
    else:
        env.root[first] = eval_expr(args[1], env)
    return None

@macro
def scheme_if(args, env):
    # Unpack the args
    (cond, conseq, alt) = args[:3]
    if eval_expr(cond, env):
        return eval_expr(conseq, env)
    else:
        return eval_expr(alt, env)

@macro
def scheme_quote(args, env):
    # Return the args[0] unevaluated
    return args[0]

@macro
def scheme_lambda(args, env):
    def inner(*params):
        for k,v in zip(args[0], params):
            # Evaluate each input param
            # With the bound environment
            # Of the closure, and add these
            # Params to the closure's environment
            env[k] = v
        # Then this function should evaluate the body(s) and return the last
        return [eval_expr(b, env) for b in args[1:]][-1]
    # Return the inner function
    return inner

@macro
def scheme_let(args, env):
    # If the first is the proc-id
    if not isinstance(args[0], list):
        e = [args[0]]
        e.extend([a[1] for a in args[1]])
        # Create a new "lambda" function
        # With the ids bound as parameter names
        # We only update the env once all bindings are eval'd
        def inner(*params):
            vals = dict()
            for k,v in zip([a[0] for a in args[1]], params):
                vals[k] = v
            # Now update the environment with the bindings
            env.update(vals)
            # Then evaluate the bodys
            return [eval_expr(b, env) for b in args[2:]][-1]
        # Bind inner
        env[args[0]] = inner

        # Execute the call 'e'
        return eval_expr(e, env)

    else:
        # Add all vals to a local dictionary
        # Then evaluate the bodys given the env
        vals = dict()
        for k,v in args[0]:
            vals[k] = eval_expr(v, env)
        # Update the env
        env.update(vals)
        return [eval_expr(b, env) for b in args[1:]][-1]

@macro
def scheme_let_star(args, env):
    for k,v in args[0]:
        env[k] = eval_expr(v, env)
    # Update the env
    return [eval_expr(b, env) for b in args[1:]][-1]

@macro
def scheme_begin(args, env):
    return [eval_expr(b, env) for b in args][-1]

# I/O
def scheme_display(s, f=stdout):
    f.write(str(to_scheme(s)).strip('"')+'\n')
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
    """Create a socket object. Only supports IPV4."""
    host = host.strip('"')
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind((host, port))
    return sock

def scheme_socket_close(sock):
    """Closes a socket object."""
    sock.close()
    return None

def scheme_socket_accept(sock):
    """Listens for a socket connection. Returns a socket object for the connection after accepting."""
    sock.listen(1)
    (conn, addr) = sock.accept()
    return conn

def scheme_socket_readline(conn):
    """Listens for a newline terminated string on the connection object and returns it. Strips the terminating newline."""
    chunks = list()
    while True:
        chunk = conn.recv(4096)
        chunks.append(chunk)
        if chunk.endswith('\n'):
            break
    return ''.join(chunks).rstrip('\n')

def scheme_socket_read(conn, num_bytes):
    """Reads num_bytes bytes from socket object conn. Returns the bytes as a string."""
    chunks = list()
    while True:
        chunks.append(conn.recv(1))
        if len(chunks) == num_bytes:
            break
    return ''.join(chunks)

def scheme_socket_write(conn, s):
    """Writes argument s to the conn socket object."""
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
    """Joins each string in the strs list together with the separator.
    Returns a string consisting of the concatenation of the new list."""
    return sep.join(strs)

# List manipulation
def scheme_list(*args):
    return args

def scheme_length(l):
    return len(l)

def scheme_car(l):
    """Returns the first element of the list. Error if l is empty."""
    return l[0]

def scheme_cdr(l):
    """Returns the sublist obtained by removing the first element."""
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
    """Applies function func to each element of vals and returns a new list consisting of the
    output of each execution of procedure proc."""
    return [func(v) for v in vals]

def scheme_for_each(func, vals):
    """Applies function func to each elementent of val. The return values of func are ignored."""
    for v in vals:
        func(v)
    return None

# Threads
def scheme_thread_create(expr):
    """Creates a new thread (but does not execute it)
    with the given expression. Returns the thread object."""
    t = threading.Thread(target=expr)
    t.start()
    return t

def scheme_thread_run(t):
    """Executes the function associated with the thread object t."""
    t.run()
    return None

def scheme_thread_join(t):
    """Blocks the caller until the callee thread terminates."""
    t.join()
    return None

def scheme_thread_sleep(dur):
    """Sleeps this thread for the given duration dur."""
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
        self.root = None

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
    env.env = { 'if'                : scheme_if,
                'quote'             : scheme_quote,
                'define'            : scheme_define,
                'lambda'            : scheme_lambda,
                'let'               : scheme_let,
                'let*'              : scheme_let_star,
                'begin'             : scheme_begin,
                'void'              : lambda *args: None,
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

def do_apply(clo, args, env):
    # Is the closure a macro?
    # Macros should always be provided the environment
    # And should not evaluate the args
    if hasattr(clo, 'macro'):
        return clo(args, env)
    else:
        # Evaluate the args and pass to the closure
        # We need to unpack the arguments from the evaluated list
        return clo(*[eval_expr(a, env) for a in args])

def eval_expr(expr, env=global_env):
    local_env = Env()
    local_env.parent = env
    local_env.root = global_env

    if expr == 'genv':
        pprint(global_env.env)
    elif isinstance(expr, list):
        clo = eval_expr(expr[0], env)
        return do_apply(clo, expr[1:], local_env)
    # Any non-enclosed string is a symbol that should
    # Be in the environment
    elif isinstance(expr, str) and \
        not (expr.startswith('"') and expr.endswith('"')):
            return local_env.find(expr)
    else:
        return expr

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
                print_exc()
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
