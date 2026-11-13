import re, sys
L = open(sys.argv[1], encoding='utf8').read().split('\n')
CITE = re.compile(r'\\cite\{([^}]*)\}')
def keys(a, b):
    s = '\n'.join(L[a-1:b])
    return [k.strip() for g in CITE.findall(s) for k in g.split(',')]
tax = set(keys(1204, 1262))
T = {'T2': keys(427, 509), 'T3': keys(591, 641), 'T4': keys(721, 751), 'T5': keys(808, 848)}
sect = set(k for v in T.values() for k in v)
fi = set(keys(1149, 1180)); fu = set(keys(877, 887)); rep = set(keys(1051, 1116))
print('taxonomy unique', len(tax))
print('section tables unique', len(sect))
print('in section tables, not taxonomy:', sorted(sect - tax))
print('in reported tab, not taxonomy:', sorted(rep - tax))
print('FI bench not in taxonomy:', len(fi - tax), sorted(fi - tax))
print('FU bench not in taxonomy:', sorted(fu - tax))
body = '\n'.join(L[510:913])
named = set(k.strip() for g in re.findall(r'\\cite\{([^}]*)\}\s*\(', body) for k in g.split(','))
print('body "cite (Name)" keys not in taxonomy:', sorted(named - tax))
print('union taxonomy+named+FI+FU bench:', len(tax | named | fi | fu))
