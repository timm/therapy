#!/usr/bin/env python3
"""
Name:
    there.py : learn how to change, for the better

Version:
    0.2

Usage:
    there [options]

Options:

    -h        Help.
    -v        Verbose.
    -r=f      Function to run.
    -s=n      Set random number seed [default: 1].
    -k=n      Speed in knots [default: 10].

Examples:

    - Installation: `sh INSTALL.md`
    - Unit tests. 'pytest.py  there.py'
    - One Unit test. `pytest.py -s -k tion1 there.py`
    - Continual tests: `rerun 'pytest there.py'`
    - Documentation: `sh DOC.md`
    - Add some shell tricks: `sh SH.md`

Notes:
    Simplest to tricky-est, this code divides
    into `OTHER`,`BINS`,`TABLE`.

    - `OTHER` contains misc  utilities.
    - `ROW` manages sets of rows.
    - `BINS` does discretization.

Author:
   Tim Menzies
   timm@ieee.org
   http://menzies.us

Copyright:
   (c) 2020 Tim Menzies,
   MIT license,
   https://opensource.org/licenses/MIT

"""

from collections import defaultdict
import re
import sys
import math
import copy
import bisect
import pprint
from docopt import docopt
from random import random, seed, choice
from random import shuffle as rshuffle


# ---------------------------------------------
# Misc, lib functions
def opt(d, **types):
  """Coerce dictionaries into simple keys
    random.seed(1)
  whose values are of known `types`."""
  d = {re.sub(r"^[-]+", "", k): v for k, v in d.items()}
  for k, f in types.items():
    d[k] = f(d[k])
  return o(**d)


def ako(x, c): return isinstace(x, c)
def same(x): return x
def first(a): return a[0]
def last(a): return a[-1]
def shuffle(a): rshuffle(a); return a


class o:
  """LIB: Class that can pretty print; that
  can `inc`rement and `__add__` their values."""
  def __init__(i, **d):
    i.__dict__.update(**d)

  def __repr__(i):
    "Pretty print. Hide private keys (those starting in `_`)"
    def dicts(x, seen=None):
      if isinstance(x, (tuple, list)):
        return [dicts(v, seen) for v in i]
      if isinstance(x, dict):
        return {k: dicts(x[k], seen)
                for k in x if str(k)[0] != "_"}
      if isinstance(x, o):
        seen = seen or {}
        j = id(x) % 128021  # ids are LONG; show them shorter.
        if x in seen:
          return f"#:{j}"
        seen[x] = x
        d = dicts(x.__dict__, seen)
        d["#"] = j
        return d
      return x
    # -----------------------
    return re.sub(r"'", ' ',
                  pprint.pformat(dicts(i.__dict__), compact=True))


class Col(o):
  "Summarize columns. Ignore '?' unknown values."
  def __init__(i, pos=0, txt="", w=1, all=[]):
    i.n, i.w, i.pos, i.txt = 0, w, pos, txt
    i.also()
    [i + x for x in all]

  def __add__(i, x):
    if x == "?":
      return x
    i.n += 1
    i.add(x)
    return x

  def norm(i, x):
    if x == "?":
      return x
    return i.norm1(x)

  def norm1(i, x): return x

  def dist(i, x, y):
    return 1 if x == "?" and y == "?" else i.dist1(x, y)


class Num(Col):
  "Summarize numeric columns"
  def also(i, most=sys.maxsize):
    i.mu, i.m2, i.sd, i.lo, i.hi = 0, 0, 0, most, -most

  def add(i, x):
    i.lo = min(x, i.lo)
    i.hi = max(x, i.hi)
    d = x - i.mu
    i.mu += d/i.n
    i.m2 += d*(x - i.mu)
    if i.m2 < 0:
      i.sd = 0
    elif i.n <= 1:
      i.sd = 0
    else:
      i.sd = (i.m2/(i.n-1))**0.5

  def dist1(i, x, y):
    if x == "?":
      y = i.norm(y)
      x = 0 if y > 0.5 else 1
    elif y == "?":
      x = i.norm(x)
      y = 0 if x > 0.5 else 1
    else:
      x, y = i.norm(x), i.norm(y)
    return abs(x-y)

  def like(i, x, *_):
    v = i.sd**2 + 10**-64
    nom = math.e**(-1*(x-i.mu)**2/(2*v)) + 10**-64
    denom = (2*math.pi*v)**.5
    return nom/(denom + 10**-64)


class Sym(Col):
  "Summarize symbolic columns"
  def also(i):
    i.seen, i.most, i.mode = {}, 0, None

  def add(i, x):
    i.seen[x] = i.seen.get(x, 0) + 1
    if i.seen[x] > i.most:
      i.most, i.mode = i.seen[x], x

  def dist1(i, x, y):
    return 0 if x == y else 1

  def like(i, x, prior=1, m=1):
    return (i.seen.get(x, 0) + m*prior)/(i.n + m)


class Row(o):
  """
  Holds one example from a set of `rows`
  in 'cells' (and, if the row has been descretized,
  in 'bins').
  """
  def __init__(i, rows, cells):
    i._rows = rows
    i.cells = cells
    i.bins = cells[:]
    i.seen = False
    i.dom = 0

  def __getitem__(i, k):
    return i.cells[k]

  def better(i, j):
    c = i._rows.cols
    s1, s2, n = 0, 0, len(c.y) + 0.0001
    for col in c.y:
      x = i.bins[col.pos]
      y = j.bins[col.pos]
      s1 -= math.e**(col.w * (x - y) / n)
      s2 -= math.e**(col.w * (y - x) / n)
    return s1 / n < s2 / n

  def dist(i, j, what="x"):
    d, n = 0, 0
    for col in i._rows.cols[what]:
      a, b = i[col.pos], j[col.pos]
      n += 1
      inc = col.dist(a, b)
      d += inc ** 2
    return (d / (n + 0.001))**0.5

  def status(i):
    return [i[col.pos] for col in i._rows.cols.y]


class Rows(o):
  """
  Holds many examples in `rows`.  Also, `cols` stores
  type descriptions for each column (and `cols` is built from the
  names in the first row).
  """
  def __init__(i, src=None):
    """
    Create from `src`, which could be a list,
    a `.csv` file name, or a string.
    """
    i.all = []
    i.cols = o(all=[], names={}, klass=None,
               x=[], y=[], syms=[], nums=[])
    if src:
      [i.add(row) for row in csv(src)]

  def clone(i, all=[]):
    tmp = Rows()
    tmp.header([col.txt for col in i.cols.all])
    [tmp.row(one) for one in all]
    return tmp

  def add(i, row):
    "The first `row` goes to the header. All the rest got to `rows`."
    i.row(row) if i.cols.all else i.header(row)

  ch = o(klass="!", num="$",
         less="<", more=">", skip="?",
         nums="><$", goal="<>!,")

  def header(i, lst):
    """
    Using the magic characters from `Rows.ch`, divide the columns
    into the symbols, the numbers, the x cols, the y cols, the
    klass col. Also, store them all in the `all` list.
    """
    c, ch = i.cols, Rows.ch
    c.klass = -1
    for pos, txt in enumerate(lst):
      w = -1 if ch.less in txt else 1
      col = (Num if txt[0] in ch.nums else Sym)(pos, txt, w)
      (c.nums if txt[0] in ch.nums else c.syms).append(col)
      (c.y if txt[0] in ch.goal else c.x).append(col)
      if ch.klass in txt:
        c.klass = col
      c.all += [col]
      i.cols.names[txt] = col

  def row(i, z):
    "add a new row"
    z = z.cells if isinstance(z, Row) else z
    [col + val for col, val in zip(i.cols.all, z)]
    i.all += [Row(i, z)]

  def bins(i, goal=None, cohen=.2):
    """
    Divide ranges into  ranges that best select for `goal`.  If
    `goal=None` then just divide into sqrt(N) bins, that differ
    by more than a small amount (at least `.2*sd`).
    """
    def apply2Numerics(lst, x):
      if x == "?":
        return x
      for pos, bin in enumerate(lst):
        if x < bin.xlo:
          break
        if bin.xlo <= x < bin.xhi:
          break
      return round((pos + 1) / len(lst), 2)
    # ----------------
    bins = {}
    for col in i.cols.nums:
      x = col.pos
      bins[x] = Bins.nums(i.all, x=x, goal=goal,
                          cohen=cohen, y=i.cols.klass.pos)
      for row in i.all:
        old = row.bins[x]
        new = apply2Numerics(bins[x], row[x])
        row.bins[x] = new
    for col in i.cols.syms:
      x = col.pos
      bins[x] = Bins.syms(i.all, x=x, goal=goal,
                          y=i.cols.klass.pos)
    return bins

  def like(i, row, n, m, k, nh):
    prior = (len(i.all) + k) / (n + k*nh)
    out = math.log(prior)
    for col in i.cols.x:
      val = row[col.pos]
      if val != "?":
        inc = col.like(val, prior, m)
        out += math.log(inc)
    return out


class Bin(o):
  """A `bin` is a core data structure in DUO. It
  runs from some `lo` to `hi` value in a column, It is
  associated with some `ys` values. Some bins have higher
  `val`ue than others (i.e. better predict for any
  known goal."""
  def __init__(i, z="__alll__", x=0):
    i.xlo = i.xhi = z
    i.x, i.val = x, 0
    i.ys = {}

  def selects(i, row):
    """Bin`s know the `x` index of the column
    they come from (so `bin`s can be used to select rows
    whose `x` values fall in between `lo` and `hi`."""
    tmp = row[i.x]
    return tmp != "?" and i.xlo <= row[i.x] <= row[i.xhi]

  def score(i, all, e=0.00001):
    "Score a bin by prob*support that it selects for the goal."
    yes = i.ys.get(1, 0) / (all.ys.get(1, 0) + e)
    no = i.ys.get(0, 0) / (all.ys.get(0, 0) + e)
    tmp = round(yes**2 / (yes + no + e), 3)
    i.val = tmp if tmp > 0.01 else 0
    return i

  def __add__(i, j):
    "Add together the numeric values in `i` and `j`."
    k = Bin(x=i.x)
    k.xlo, k.xhi = i.xlo, j.xhi
    for x, v in i.ys.items():
      k.ys[x] = v
    for x, v in j.ys.items():
      k.ys[x] = v + k.ys.get(x, 0)
    return k

  def inc(i, y, want):
    k = y == want
    i.ys[k] = i.ys.get(k, 0) + 1


class Bins:
  "Bins is a farcade holding code to manage `bin`s."
  def syms(lst, x=0, y=-1, goal=None):
    "Return bins for columns of symbols."
    all = Bin(x=x)
    bins = {}
    for z in lst:
      xx, yy = z[x], z[y]
      if xx != "?":
        if xx not in bins:
          bins[xx] = Bin(xx, x)
        now = bins[xx]
        now.inc(yy, goal)
        all.inc(yy, goal)
    return [bin.score(all) for bin in bins.values()]

  def nums(lst, x=0, y=-1, goal=None, cohen=.3,
           enough=.5, trivial=.05):
    """
    Return bins for columns of numbers. Combine two bins if
    they are separated by too small amount or if
    they predict poorly for the goal.
    """
    def split():
      xlo, bins, n = 0, [Bin(0, x)], len(lst)**enough
      while n < 4 and n < len(lst) / 2:
        n *= 1.2
      for xhi, z in enumerate(lst):
        xx, yy = z[x], z[y]
        if xhi - xlo >= n:  # split when big enough
          if len(lst) - xhi >= n:  # split when enough remains after
            if xx != lst[xhi - 1][x]:  # split when values differ
              bins += [Bin(xhi, x)]
              xlo = xhi
        now = bins[-1]
        now.xhi = xhi + 1
        all.xhi = xhi + 1
        now.inc(yy, goal)
        all.inc(yy, goal)
      return [bin.score(all) for bin in bins]

    def merge(bins):
      j, tmp = 0, []
      while j < len(bins):
        a = bins[j]
        if j < len(bins) - 1:
          b = bins[j + 1]
          ab = (a + b).score(all)
          tooLittleDifference = (mid(b) - mid(a)) < cohen
          notBetterForGoal = goal and ab.val >= a.val and ab.val >= b.val
          if tooLittleDifference or notBetterForGoal:
            a = ab
            j += 1
        tmp += [a]
        j += 1
      return bins if len(tmp) == len(bins) else merge(tmp)

    def mid(z): return (n(z.xlo) + n(z.xhi)) / 2
    def per(z=0.5): return lst[int(len(lst) * z)][x]
    def n(z): return lst[min(len(lst) - 1, z)][x]
    def finalize(z): z.xlo, z.xhi = n(z.xlo), n(z.xhi); return z
    # --------------------------------------------------------------
    lst = sorted((z for z in lst if z[x] != "?"), key=lambda z: z[x])
    all = Bin(0, x)
    cohen = cohen * (per(.9) - per(.1)) / 2.54
    return [finalize(bin) for bin in merge(split())]


class Abcd:
  """Track set of actual and predictions, report precsion, accuracy,
  false alarm, recall,...
  """
  def __init__(i, db="all", rx="all"):
    i.db = db
    i.rx = rx
    i.yes = i.no = 0
    i.known = {}
    i.a = {}
    i.b = {}
    i.c = {}
    i.d = {}
    i.all = {}

  def __call__(i, actual, predict):
    i.knowns(actual)
    i.knowns(predict)
    if actual == predict:
      i.yes += 1
    else:
      i.no += 1
    for x in i.known:
      if actual == x:
        if predict == actual:
          i.d[x] += 1
        else:
          i.b[x] += 1
      else:
        if predict == x:
          i.c[x] += 1
        else:
          i.a[x] += 1

  def knowns(i, x):
    if not x in i.known:
      i.known[x] = i.a[x] = i.b[x] = i.c[x] = i.d[x] = 0.0
    i.known[x] += 1
    if (i.known[x] == 1):
      i.a[x] = i.yes + i.no

  def report(i):
    print("")
    print('{0:20s} {1:10s} {2:3s}  {3:3s} {4:3s} {5:3s} {6:3s} {7:3s} {8:3s} {9:3s} {10:3s} {11:3s} {12:3s} {13:10s}'.format(
        "db", "rx", "n", "a", "b", "c", "d", "acc", "pd", "pf", "prec", "f", "g", "class"))
    print('-'*85)
    def p(y): return int(100*y + 0.5)
    def n(y): return int(y)
    pd = pf = pn = prec = g = f = acc = 0
    order = sorted([(-1*(i.b[k] + i.d[k]), k)
                    for k in i.known])
    for _, x in order:
      a = i.a[x]
      b = i.b[x]
      c = i.c[x]
      d = i.d[x]
      if (b+d):
        pd = 1.0*d / (b+d)
      if (a+c):
        pf = 1.0*c / (a+c)
      if (a+c):
        pn = 1.0*(b+d) / (a+c)
      if (c+d):
        prec = 1.0*d / (c+d)
      if (1-pf+pd):
        g = 2.0*(1-pf)*pd / (1-pf+pd)
      if (prec+pd):
        f = 2.0*prec*pd/(prec+pd)
      if (i.yes + i.no):
        acc = 1.0*i.yes/(i.yes+i.no)
      i.all[x] = o(pd=pd, pf=pf, prec=prec, g=g, f=f, acc=acc)
      print('{0:20s} {1:10s} {2:3d} {3:3d} {4:3d} {5:3d} {6:3d} {7:3d} {8:3d} {9:3d} {10:3d} {11:3d} {12:3d} {13:10s}'.format(
          i.db, i.rx, n(b + d), n(a), n(b), n(c), n(d), p(acc), p(pd), p(pf), p(prec), p(f), p(g), x))


class Seen(o):
  def __init__(i,  rows, m=2, k=1):
    i.rows, i.m, i.k = rows, m, k
    i.ys, i.n = {}, 0

  def train(i, row):
    y = row[i.rows.cols.klass.pos]
    if y not in i.ys:
      i.ys[y] = i.rows.clone()
    i.n += 1
    i.ys[y].row(row)

  def guess(i, row):
    all, ybest, most = [], None, -10**64
    for y in i.ys:
      tmp = i.ys[y].like(row, i.n, i.m, i.k, len(i.ys))
      all += [(tmp, row)]
      if tmp > most:
        ybest, most = y, tmp
    return ybest, all

  def uncertain(i, rows):
    all = []
    for row in rows:
      tmp = i.guess(row)[1]
      two, one = tmp[-2][0], tmp[-1][0]
      n1 = doubt = 1 - abs(one-two)
      n2 = strength = one
      all += [((n1**2 + n1**2)**0.5, n1, n2, row)]
    return sorted(all, key=first)


def csv(src=None, f=sys.stdin):
  """Read from stdio or file or string or list.  Kill whitespace or
  comments. Coerce number strings to numbers."Ignore columns if,
  on line one, the name contains '?'."""
  def items(z):
    for y in z:
      yield y

  def strings(z):
    for y in z.splitlines():
      yield y

  def csv(z):
    with open(z) as fp:
      for y in fp:
        yield y

  def rows(z):
    for y in f(z):
      if isinstance(y, str):
        y = re.sub(r'([\n\t\r ]|#.*)', '', y).strip()
        if y:
          yield y.split(",")
      else:
        yield y

  def floats(a): return a if a == "?" else float(a)

  def nums(z):
    funs, num = None, Rows.ch.nums
    for a in z:
      if funs:
        yield [fun(a1) for fun, a1 in zip(funs, a)]
      else:
        funs = [floats if a1[0] in num else str for a1 in a]
        yield a

  def cols(src, todo=None):
    for a in src:
      todo = todo or [n for n, a1 in enumerate(a) if "?" not in a1]
      yield [a[n] for n in todo]

  if src:
    if isinstance(src, (list, tuple)):
      f = items
    elif isinstance(src, str):
      if src[-3:] == 'csv':
        f = csv
      else:
        f = strings
  for row in nums(cols(rows(src))):
    yield row
