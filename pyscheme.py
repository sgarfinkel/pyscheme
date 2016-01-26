from sys import argv

# This will go in a separate module eventually
# Along with the dictionary of classes
class SchemeDisplay:
    def __init__(self):
        self.vals = list()
    # Set the val
    def add_val(self, val):
        if val.startswith('"') and val.endswith('"'):
            self.vals.append(val.lstrip('"').rstrip('"'))
        else:
            raise(Exception)
    # Execute
    def execute(self):
        print self.vals[0]
        return None

func_dict = {'display': SchemeDisplay}

# This is the public tokenizer, which is what should be called
# Wrapper around the private method which removes a dimension
# And returns only the tokens, no indices
def tokenize(text):
    (tokens, k) = _tokenize(text, 0)
    return tokens[0]

# Recursively tokenize
# This is a private method
def _tokenize(text, i):
    symbol_table = list()
    in_string = False
    # Keep every symbol as a list
    # Join and append when we hit whitespace or a parenthesis
    # Unless the symbol is a string
    symbol = list()
    while i < len(text):
        char = text[i]
        # If we hit the start of a new function
        # Recursively call to 'jump' a dimension
        if char == '(' and not in_string:
            (tokens, k) = _tokenize(text, i+1)
            # Append this next level to our symbol_table
            symbol_table.append(tokens)
            # Set i
            i = k
        # When we hit the end of a function, increment i
        # For returning, and break the loop
        if char == ')' and not in_string:
            i += 1
            break
        # Skip whitespace and newlines unless we're in a string
        if (char == ' ' or char == '\n') and not in_string:
            # Add what we had as a new symbol
            symbol_table.append(''.join(symbol))
            # And clear existing symbol
            symbol = list()
            # Skip everything else in this iteration
            i += 1
            continue
        symbol.append(char)
        # A quotation mark starts or ends a string
        # If you're in a string, append this as a new symbol
        # Flip the boolean in_string
        if char == '"':
            if in_string:
                symbol_table.append(''.join(symbol))
                symbol = list()
            in_string = not in_string
        # Increment i
        i += 1

    # Return our symbol table
    return (symbol_table, i)

# This recursively parses our tokens
def parse_tokens(tokens):
    func = None
    for i, t in enumerate(tokens):
        if i == 0:
            func = func_dict[t]()
        elif isinstance(t, list):
            o = parse_tokens(t)
            func.add_val(o)
        else:
            func.add_val(t)
    o = func.execute()
    if o:
        return o


# Main line
if __name__ == '__main__':
    with open(argv[1]) as f:
        tokens = tokenize(f.read())
        parse_tokens(tokens)