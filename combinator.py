#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © 2025 Ulrich Drepper <drepper@akkadia.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Evaluation and reduction of lambda expression."""
from __future__ import annotations

import argparse
import functools
import itertools
import os
import sys
from typing import cast, ClassVar, Dict, List, override, Optional, Set, Tuple

from colored import Fore, Style


# These are the characters accepted and used s variable names.  The list
# could be extended here and everything else should just work.  But it
# should be noted that
# a) uppercase characters are used in the names of known combinators
# b) some greek letters (both lowercase like λ and π and uppercase like
#    Φ and Ψ
# are used for non-variables and conflicts would be fatal.
VARIABLE_NAMES = 'abcdefghijklmnopqrstuvwxyz'


class Naming:
  """This is an object encapsulating the predictable generation of distinct
  variable names."""
  def __init__(self, avoid: Set[str]):
    self.next = VARIABLE_NAMES[0]
    self.known = {}
    self.avoid = avoid
    if self.next in self.avoid:
      self.next_var()

  def get(self, v: Var) -> str:
    """Get the next variable name."""
    if v.id not in self.known:
      assert self.next in VARIABLE_NAMES, 'too many variables'
      try:
        self.known[v.id] = self.next
        self.next_var()
      except IndexError:
        # set to an invalid value
        self.next = '_'
    return self.known[v.id]

  def contains(self, v: Var) -> bool:
    """Return true if given id is known."""
    return v.id in self.known

  def add(self, other: Naming) -> None:
    """Add definitions from other variable set to own."""
    assert self.avoid == other.avoid, "avoid sets must be equal"
    if other.known:
      self.next = VARIABLE_NAMES[max(VARIABLE_NAMES.index(self.next), VARIABLE_NAMES.index(other.next))]
      assert not set(self.known.keys()) & set(other.known.keys())
      self.known = self.known | other.known

  def next_var(self) -> None:
    """Determine the next variable name, taking the names to avoid into account."""
    while True:
      idx = VARIABLE_NAMES.index(self.next) + 1
      self.next = VARIABLE_NAMES[idx] if idx < len(VARIABLE_NAMES) else ''
      if self.next not in self.avoid:
        return


# List of combinators, mostly taken from
#   https://www.angelfire.com/tx4/cus/combinator/birds.html
# These correspond to the Smullyan's "To Mock a Mockingbird".
# The list can be extended.  The names are used in translation from the
# input and when printing the result.
KNOWN_COMBINATORS = {
  'B': 'λabc.a(bc)',
  'B₁': 'λabcd.a(bcd)',
  'B₂': 'λabcde.a(bcde)',
  'B₃': 'λabcd.a(b(cd))',
  'C': 'λabc.acb',
  'C*': 'λabcd.abdc',
  'C**': 'λabcde.abced',
  'D': 'λabcd.ab(cd)',
  'D₁': 'λabcde.abc(de)',
  'D₂': 'λabcde.a(bc)(de)',
  'E': 'λabcde.ab(cde)',
  'Ê': 'λabcdefg.a(bcd)(efg)',
  'F': 'λabc.cba',
  'F*': 'λabcd.adcb',
  'F**': 'λabcde.abedc',
  'G': 'λabcd.ad(bc)',
  'H': 'λabc.abcb',
  'I': 'λa.a',
  'I*': 'λab.ab',
  'I**': 'λabc.abc',
  'J': 'λabcd.ab(adc)',
  'K': 'λab.a',
  'L': 'λab.a(bb)',
  'M': 'λa.aa',
  'M₂': 'λab.ab(ab)',
  'O': 'λab.b(ab)',
  'π': 'λab.b',
  'Φ': 'λabcd.a(bd)(cd)',
  'Φ₁': 'λabcde.a(bde)(cde)',
  'Ψ': 'λabcd.a(bc)(bd)',
  'Q': 'λabc.b(ac)',
  'Q₁': 'λabc.a(cb)',
  'Q₂': 'λabc.b(ca)',
  'Q₃': 'λabc.c(ab)',
  'Q₄': 'λabc.c(ba)',
  'R': 'λabc.bca',
  'R*': 'λabcd.acdb',
  'R**': 'λabcde.abdec',
  'S': 'λabc.ac(bc)',
  'T': 'λab.ba',
  'U': 'λab.b(aab)',
  'V': 'λabc.cab',
  'V*': 'λabcd.adbc',
  'V**': 'λabcde.abecd',
  'W': 'λab.abb',
  'W¹': 'λab.baa',
  'W*': 'λabc.abcc',
  'W**': 'λabcd.abcdd',
}


def remove_braces(s: str) -> str:
  """Return argument without enclosing parenthesis or the string itself."""
  return s[1:-1] if s and s[0] == '(' and s[-1] == ')' else s


class Obj:
  """Base class for node in the graph representation of a lambda expression."""
  def is_free_in_context(self, v: Var) -> bool: # pylint: disable=unused-argument
    """Test whether this is an object for a free variable in the context.  This is the generic
    implementation."""
    return True

  def __str__(self) -> str:
    raise NotImplementedError('__str__ called for Obj')

  def fmt(self, varmap: Naming, highlight: bool) -> str:
    """Format an expression as a string.  This pure virtual version must never
    be called."""
    raise NotImplementedError('fmt called for Obj')

  def replace(self, v: Var, expr: Obj) -> Obj: # pylint: disable=unused-argument
    """Return the expression with the given variable replaced by the expression."""
    return self

  def duplicate(self) -> Obj:
    """Duplicate the object."""
    return self

  def recombine(self) -> Obj:
    """Recombine combinators."""
    return self

  def rmatch(self, other: Obj, var_map: Dict[Var, Obj]) -> bool: # pylint: disable=unused-argument
    """Determine whether the expression matches OTHER considering variable renaming in VAR_MAP."""
    return self == other

  def collect_free_vars(self) -> Set[str]:
    """Return a set of all the variables that are free for the entire expression."""
    return set()


class Var(Obj):
  """Object to represent a variable in the lambda expression graph.  This implements
  the de Bruijn notation by representing each new variable with a unique number."""
  varcnt: ClassVar[int] = 1
  color: ClassVar[str] = Fore.rgb(242, 185, 45)
  color_free: ClassVar[str] = Fore.rgb(255, 64, 23)

  def __init__(self, freename: Optional[str] = None):
    self.id = Var.varcnt
    self.freename = freename
    Var.varcnt += 1

  @override
  def is_free_in_context(self, v: Var) -> bool:
    """Test whether this is a free variable.  If we come here and this is the variable
    we are looking for it is indeed free in the context."""
    return self.id != v.id

  @override
  def __str__(self):
    return f'{{var {self.id}}}'

  @override
  def fmt(self, varmap: Naming, highlight: bool) -> str:
    res = self.freename or varmap.get(self)
    return f'{Var.color_free if self.freename else Var.color}{res}{Style.reset}' if highlight else res

  @override
  def replace(self, v: Var, expr: Obj) -> Obj:
    return expr.duplicate() if v.id == self.id else self

  @override
  def rmatch(self, other: Obj, var_map: Dict[Var, Obj]) -> bool:
    if not isinstance(other, Var):
      return False
    if self.id == other.id or var_map.get(other) == self:
      return True
    if isinstance(var_map.get(other), Empty) and self not in var_map.values():
      var_map[other] = self
      return True
    return False

  @override
  def collect_free_vars(self) -> Set[str]:
    return set(self.freename) if self.freename else set()


class Empty(Obj):
  """Object returned to indicate errors when parsing lambda expressions and other situations when a non-existing
  object needs to be represented."""
  @override
  def __str__(self):
    return '{}'

  @override
  def fmt(self, varmap: Naming, highlight: bool) -> str:
    # This is a filler.  An object of this type should never really be used.
    return '○'


class Constant(Obj):
  """Object to represent constants in the lambda expression graph."""
  def __init__(self, name: str):
    self.name = name

  @override
  def __str__(self):
    return f'{{const {self.name}}}'

  @override
  def fmt(self, varmap: Naming, highlight: bool) -> str:
    return f'{self.name} '



class Combinator(Obj):
  """Object to represent a recombined combinator."""
  color: ClassVar[str] = Fore.rgb(255, 0, 163)

  def __init__(self, combinator: str, arguments: Optional[List[Obj]] = None):
    self.combinator = combinator
    self.arguments = [] if arguments is None else arguments

  @override
  def is_free_in_context(self, v: Var) -> bool:
    return True

  @override
  def __str__(self):
    if self.arguments:
      return f'{{{self.combinator} {' '.join([str(a) for a in self.arguments])}}}'
    return self.combinator

  @override
  def fmt(self, varmap: Naming, highlight: bool) -> str:
    combres = f'{Combinator.color}{self.combinator}{Style.reset}' if highlight else self.combinator
    if self.arguments:
      combres += ' ' + ' '.join([a.fmt(varmap, highlight) for a in self.arguments])
    return combres


class Application(Obj):
  """Object to represent the application (call) to a function in the lambda
  expression graph."""
  def __init__(self, ls: List[Obj]):
    assert len(ls) >= 2
    self.code = (ls[0].code + ls[1:]) if isinstance(ls[0], Application) else ls
    assert self.code

  @override
  def is_free_in_context(self, v: Var) -> bool:
    return all(e.is_free_in_context(v) for e in self.code)

  @override
  def __str__(self):
    return f'{{App {' '.join([str(a) for a in self.code])}}}'

  @override
  def fmt(self, varmap: Naming, highlight: bool) -> str:
    return f'({''.join([a.fmt(varmap, highlight) for a in self.code]).rstrip()})'

  @override
  def replace(self, v: Var, expr: Obj) -> Obj:
    return apply([e.replace(v, expr) for e in self.code])

  @override
  def duplicate(self) -> Obj:
    return Application([e.duplicate() for e in self.code])

  @override
  def recombine(self) -> Obj:
    return Application([r.recombine() for r in self.code])

  @override
  def rmatch(self, other: Obj, var_map: Dict[Var, Obj]) -> bool:
    return isinstance(other, Application) and len(self.code) == len(other.code) and all(a.rmatch(b, var_map) for a, b in zip(self.code, other.code))

  @override
  def collect_free_vars(self) -> Set[str]:
    return set().union(*[e.collect_free_vars() for e in self.code])

  def beta(self) -> Obj:
    """Perform beta reduction on the given application.  This is called on a freshly
    created object but the reduction cannot be performed in the constructor because
    the result of the beta reduction can be something other than an application."""
    if not isinstance(self.code[0], Lambda):
      return self
    la = cast(Lambda, self.code[0])
    r = la.code.replace(la.params[0], self.code[1])
    if len(la.params) > 1:
      r = newlambda(la.params[1:], r)
    return apply([r] + self.code[2:])


class Lambda(Obj):
  """Object to represent a lambda expression in the lambda expression graph."""
  color: ClassVar[str] = Fore.rgb(45, 135, 242)

  def __init__(self, params: List[Var], code: Obj):
    if not params:
      raise SyntaxError('lambda parameter list cannot be empty')
    if isinstance(code, Lambda):
      self.params = params + code.params
      self.code = code.code
    else:
      self.params = params
      self.code = code

  @override
  def is_free_in_context(self, v: Var) -> bool:
    return v not in self.params and self.code.is_free_in_context(v)

  @override
  def __str__(self):
    return f'{{lambda {' '.join([str(a) for a in self.params])}.{str(self.code)}}}'

  @override
  def fmt(self, varmap: Naming, highlight: bool) -> str:
    # It is important to process the params first to ensure correct naming of parameter variables.
    # Assign new names to the parameter variables
    nvarmap = Naming(varmap.avoid)
    paramstr = ''.join([a.fmt(nvarmap, highlight) for a in self.params])
    varmap.add(nvarmap)
    la = f'{Lambda.color}λ{Style.reset}' if highlight else 'λ'
    return f'({la}{paramstr}.{remove_braces(self.code.fmt(varmap, highlight))})'

  @override
  def replace(self, v: Var, expr: Obj) -> Obj:
    return newlambda(self.params, self.code.replace(v, expr))

  @override
  def duplicate(self) -> Obj:
    newparams = [Var() for _ in self.params]
    newcode = functools.reduce(lambda p, o: p.replace(*o), zip(self.params, newparams), self.code)
    return newlambda(newparams, newcode)

  @override
  def recombine(self) -> Obj:
    rself = Lambda(self.params, self.code.recombine())
    for comb, combstr in KNOWN_COMBINATORS.items():
      combexpr = from_string(combstr)
      assert isinstance(combexpr, Lambda)
      # Recognize the K combinator with first argument consisting only of expressions with free variables.
      if comb == 'K' and len(self.params) == 1 and len(combexpr.params) > 1 and self.code.is_free_in_context(self.params[0]):
        return Combinator(comb, [self.code])
      if len(self.params) <= len(combexpr.params):
        # We are interested in simplifications which do not introduce deeper levels of nesting.  This means
        # we do not have to perform exhaustive recursive searches.
        param_map = cast(Dict[Var, Obj], dict(itertools.zip_longest(reversed(combexpr.params), reversed(self.params), fillvalue=Empty())))
        if self.rmatch(combexpr, param_map):
          return Combinator(comb, [param_map[p] for p in combexpr.params[:len(combexpr.params)-len(self.params)]])
    return rself

  @override
  def rmatch(self, other: Obj, var_map: Dict[Var, Obj]) -> bool:
    if not isinstance(other, Lambda):
      return False
    newvar_map = var_map.copy()
    for param, newparam in zip(self.params, other.params):
      newvar_map[param] = newparam
    if self.code.rmatch(other.code, newvar_map):
      # Propagate the newly found values back to the caller.
      for k in var_map:
        if isinstance(var_map[k], Empty):
          var_map[k] = newvar_map[k]
      return True
    return False

  @override
  def collect_free_vars(self) -> Set[str]:
    return self.code.collect_free_vars()


def parse_lambda(s: str, ctx: Dict[str, Var]) -> Tuple[Obj, str]:
  """Parse the representation of a lambda definition.  Return the graph
  representation and the remainder of the string not part of the just
  parsed part."""
  assert s[0] == 'λ'
  s = s[1:].strip()
  params: List[Var] = []
  recctx = ctx.copy()
  while s:
    if s[0] == '.':
      break
    if s[0] not in VARIABLE_NAMES:
      raise SyntaxError(f'invalid λ parameters {s[0]}')
    recctx[s[0]] = Var()
    params.append(recctx[s[0]])
    s = s[1:]
  # The following is basically the loop from parse_top but with a special
  # handling of whitespaces: they terminate the body.  This is nothing
  # mandatory from general lambda expression parsing point-of-view.  It is
  # just an expectation of people using the λ… notation.  A whitespace to
  # terminate a constant is ignored.  So, an expression using a whitespace
  # in a lambda body can simply use parenthesis.
  s = s[1:].strip()
  body: List[Obj] = []
  while s:
    if s[0].isspace():
      break
    e, s = parse_one(s, recctx)
    body.append(e)
  return newlambda(params, apply(body)), s


def parse_paren(s: str, ctx: Dict[str, Var]) -> Tuple[Obj, str]:
  """Parse an expression in parenthesis.  Return the graph
  representation and the remainder of the string not part of the just
  parsed part."""
  assert s[0] == '('
  start = 1
  end = 1
  depth = 0
  while True:
    if end == len(s):
      raise SyntaxError('incomplete parenthesis')
    if s[end] == ')':
      if depth == 0:
        res = parse_top(s[start:end], ctx)
        return res, s[end + 1:]
      depth -= 1
    if s[end] == '(':
      depth += 1
    end += 1


def get_constant(s: str) -> Tuple[Obj, str]:
  """Parse a constant name.  Unlike a variable, a constant can be longer than a
  single character and therefore has to be terminated by either of the special
  characters using the string representation (λ, (, ), or .) or a whitespace.
  Return the graph representation and the remainder of the string not part of
  the just parsed part."""
  delimiters = {')', '(', '.', 'λ', '\n', '\t', '\v', '\f', ' ', '\r'}
  end_idx = 0
  while end_idx < len(s) and s[end_idx] not in delimiters:
    end_idx += 1

  token = s[:end_idx]
  # Find the longest known combinator that is a prefix of the token.
  for i in range(len(token), 0, -1):
    # If the token at this position would have one of the known suffixes it must include the latter.
    if i < end_idx and token[i] in {'*', '₁', '₂', '₃', '₄', '¹'}:
      continue
    prefix = token[:i]
    if prefix in KNOWN_COMBINATORS:
      expr = parse_top(KNOWN_COMBINATORS[prefix], {})
      return expr, s[i:].lstrip()

  # No known combinator found, treat the whole token as a constant.
  return Constant(token), s[end_idx:].lstrip()


def parse_one(s: str, ctx: Dict[str, Var]) -> Tuple[Obj, str]:
  """Toplevel function to parse a string representation of a lambda expression.
  Return the graph representation and the remainder of the string not part of
  the just parsed part."""
  match s[0]:
    case 'λ':
      return parse_lambda(s, ctx)
    case '(':
      return parse_paren(s, ctx)
    case c if c in VARIABLE_NAMES:
      return ctx[s[0]] if s[0] in ctx else Var(s[0]), s[1:]
    case c if c.isalpha():
      return get_constant(s)
    case _:
      raise SyntaxError(f'cannot parse {s}')


def newlambda(params: List[Var], code: Obj) -> Obj:
  """Create a new lambda expression using the given parametesr and body of code.
  But the function also performs η-reduction, i.e., it returns just the function
  expression (first of the application values) in case the resulting lambda would
  just apply the required parameter(s) to the application value in order."""
  if isinstance(code, Application) and len(params) < len(code.code):
    ncode = len(code.code) - len(params)
    if params == code.code[ncode:] and all(c.is_free_in_context(e) for e in params for c in code.code[:ncode]):
      return code.code[0] if ncode == 1 else Application(code.code[:ncode])
  return Lambda(params, code)


def apply(li: List[Obj]) -> Obj:
  """Create an application expression given the list of objects.  If only a
  singular expression is given it is returned, no need to wrap it into an
  application expression."""
  match len(li):
    case 0:
      res = Empty()
    case 1:
      res = li[0]
    case _:
      try:
        res = Application(li).beta()
      except RecursionError:
        res = Application(li)
  return res


def parse_top(s: str, ctx: Dict[str, Var]) -> Obj:
  """Parse a string to a lambda expression, taking one part at a time
  an creating an application expression from the parts.  Return the graph
  representation and the remainder of the string not part of the just
  parsed part."""
  s = s.strip()
  res: List[Obj] = []
  while s:
    e, s = parse_one(s, ctx)
    res.append(e)
    s = s.lstrip()
  return apply(res)


def from_string(s: str) -> Obj:
  """Parse a string to a lambda expression, taking one part at a time
  and creating an application expression from the parts.  Return the graph
  representation.  In case not all of the expression is parsed raise a
  syntax error exception."""
  return parse_top(s, {})


def to_string(expr: Obj, highlight: bool = False) -> str:
  """Return a string representation for the lambda expression graph."""
  return remove_braces(expr.recombine().fmt(Naming(expr.collect_free_vars()), highlight)).rstrip()


def handle(al: List[str], echo: bool, is_terminal: bool = False) -> int:
  """Loop over given list of strings, parse, simplify, and print the lambda
  expression."""
  ec = 0
  pr = f'{Fore.rgb(45, 242, 57)}⇒{Style.reset} ' if is_terminal else '⇒ '
  for a in al:
    if echo:
      print(a)
    try:
      print(f'{pr}{to_string(from_string(a), is_terminal)}')
    except SyntaxError as e:
      print(f'eval("{a}") failed: {e.args[0]}')
      ec = 1
    if not echo:
      print('\u2501' * (os.get_terminal_size()[0] if is_terminal else 72))
  return ec


def repl() -> int:
  """This is the REPL."""
  ec = 0
  try:
    is_terminal = sys.stdout.isatty()
    while True:
      s = input(f'{Fore.rgb(242, 45, 57)}»{Style.reset} ' if is_terminal else '» ')
      if not s:
        break
      ec = ec | handle([s], False, is_terminal)
  except EOFError:
    print('')
  return ec


def check() -> int:
  """Sanity checks.  Return error code that is used as the exit code of the process."""
  checks = [
    ('S K K', 'I'),
    ('K I', 'π'),
    ('K (S K K)', 'π'),
    ('B B', 'D'),
    ('B D', 'D₁'),
    ('B (B B)', 'D₁'),
    ('D D', 'D₂'),
    ('B B D', 'D₂'),
    ('D (B B)', 'D₂'),
    ('B B (B B)', 'D₂'),
    ('B B₁', 'E'),
    ('B (D B)', 'E'),
    ('B (B B B)', 'E'),
    ('D B', 'B₁'),
    ('D B₁', 'B₂'),
    ('B D B', 'B₃'),
    ('S(B B S)(K K)', 'C'),
    ('C I', 'T'),
    ('S(K S)K', 'B'),
    ('B₁ S B', 'Φ'),
    ('B Φ Φ', 'Φ₁'),
    ('B (Φ B S) K K', 'C'),
    ('B(S Φ C B)B', 'Ψ'),
    ('λx.NotX x', 'NotX'),
    ('B B C', 'G'),
    ('E T T E T', 'F'),
    ('B W (B C)', 'H'),
    ('B(B C)(W(B C(B(B B B))))', 'J'),
    ('C B M', 'L'),
    ('B M', 'M₂'),
    ('S I', 'O'),
    ('C B', 'Q'),
    ('B C B', 'Q₁'),
    ('C(B C B)', 'Q₂'),
    ('B T', 'Q₃'),
    ('F* B', 'Q₄'),
    ('B B T', 'R'),
    ('L O', 'U'),
    ('B C T', 'V'),
    ('C(B M R)', 'W'),
    ('C W', 'W¹'),
    ('S(S K)', 'I*'),
    ('B W', 'W*'),
    ('B C', 'C*'),
    ('B(B W)', 'W**'),
    ('B C*', 'C**'),
    ('C* C*', 'R*'),
    ('B(B B B)(B(B B B))', 'Ê'),
    ('λabcd.MMMabcd', 'MMM'),
    ('λabcd.MMMabdc', 'λabcd.MMMabdc'),
    ('λabcd.MMMabc', 'λabcd.MMMabc'),
    ('S(K e)I', 'e'),
    ('B C* R*', 'F*'),
    ('C* F*', 'V*'),
    ('B C*', 'C**'),
    ('B R*', 'R**'),
    ('B F*', 'F**'),
    ('B V*', 'V**'),
    # Simplification rules (Augustsson)
    ('S (K a) (K b)', 'K (ab)'),
    ('S (K a) I', 'a'),
    ('S (K a) b', 'B a b'),
    ('S a (K b)', 'C a b'),
    ('S (B a b) c', 'Φ a b c'),
    # ('C (B a b) c', 'C# a b c'), # where 'C# = λabcd.a(b d)c'
    ('B (a b) c', 'D a b c'),
  ]
  ec = 0
  for testinput, expected in [(key, key) for key in KNOWN_COMBINATORS] + checks:
    resexpr = from_string(testinput)
    res = to_string(resexpr)
    if res != expected:
      if expected in KNOWN_COMBINATORS:
        print(f'❌ {testinput} → {res} {resexpr} but {expected} {from_string(expected)} = {KNOWN_COMBINATORS[expected]} expected')
      else:
        print(f'❌ {testinput} → {res} {resexpr} but {expected} {from_string(expected)} expected')
      ec = 1
    else:
      print(f'✅ {testinput} → {res}')
  return ec


# def main(al: List[str]) -> None:
def main() -> None:
  """Called as main function of the program."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--check', dest='check', action='store_true')
  parser.add_argument('expression', metavar='expression', type=str, nargs='*')
  args = parser.parse_args()

  if args.check:
    # Overwrite eventual user setting
    args.tracing = False
    ec = check()
  elif args.expression:
    ec = handle(args.expression, True)
  else:
    ec = repl()
  sys.exit(ec)


if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('-t', '--tracing', dest='tracing', action='store_true')
  # parser.add_argument('--check', dest='check', action='store_true')
  # parser.add_argument('expression', metavar='expression', type=str, nargs='*')
  # args = parser.parse_args()
  # main(args.expression)
  main()
