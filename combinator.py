#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright © 2025 Ulrich Drepper <drepper@akkadia.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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
from typing import ClassVar, Self, cast, final, override

COLORS = {
    n: ""
    for n in [
        "off",
        "input_prompt",
        "var",
        "freevar",
        "combinator",
        "lambda",
        "output_prompt",
    ]
}
is_terminal = False  # pylint: disable=invalid-name


def init_terminal() -> None:
    """Initialize configuration to use terminal features."""
    from colored import Fore, Style  # pyright: ignore[reportMissingTypeStubs] pylint: disable=import-outside-toplevel

    global is_terminal  # pylint: disable=global-statement
    is_terminal = sys.stdout.isatty()
    if is_terminal:
        for k in COLORS:
            match k:
                case "input_prompt":
                    newval = Fore.rgb(242, 45, 57)
                case "output_prompt":
                    newval = Fore.rgb(45, 242, 57)
                case "var":
                    newval = Fore.rgb(242, 185, 45)
                case "freevar":
                    newval = Fore.rgb(255, 64, 23)
                case "combinator":
                    newval = Fore.rgb(255, 0, 163)
                case "lambda":
                    newval = Fore.rgb(45, 135, 242)
                case "off":
                    newval = Style.reset
                case _:
                    print(f"unhandled color {k}")
                    sys.exit(1)
            COLORS[k] = newval


# These are the characters accepted and used s variable names.  The list
# could be extended here and everything else should just work.  But it
# should be noted that
# a) uppercase characters are used in the names of known combinators
# b) some greek letters (both lowercase like λ and π and uppercase like
#    Φ and Ψ
# are used for non-variables and conflicts would be fatal.
VARIABLE_NAMES = "abcdefghijklmnopqrstuvwxyz"

# For type signature we can use the uppercase characters.
TYPE_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Naming:
    """This is an object encapsulating the predictable generation of distinct
    variable names."""

    def __init__(self, avoid: set[str]):
        self.next: str = VARIABLE_NAMES[0]
        self.known: dict[int, str] = {}
        self.avoid: set[str] = avoid
        if self.next in self.avoid:
            self.next_var()

    def get(self, v: Var) -> str:
        """Get the next variable name."""
        if v.id not in self.known:
            assert self.next in VARIABLE_NAMES, "too many variables"
            try:
                self.known[v.id] = self.next
                self.next_var()
            except IndexError:
                # set to an invalid value
                self.next = "_"
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
            self.next = VARIABLE_NAMES[idx] if idx < len(VARIABLE_NAMES) else ""
            if self.next not in self.avoid:
                return


# List of combinators, mostly taken from
#   https://www.angelfire.com/tx4/cus/combinator/birds.html
# These correspond to the Smullyan's "To Mock a Mockingbird".
# The list can be extended.  The names are used in translation from the
# input and when printing the result.
KNOWN_COMBINATORS = {
    "B": "λabc.a(bc)",
    "B₁": "λabcd.a(bcd)",
    "B₂": "λabcde.a(bcde)",
    "B₃": "λabcd.a(b(cd))",
    "C": "λabc.acb",
    "C*": "λabcd.abdc",
    "C**": "λabcde.abced",
    "C#": "λabcd.a(b d)c",
    "D": "λabcd.ab(cd)",
    "D₁": "λabcde.abc(de)",
    "D₂": "λabcde.a(bc)(de)",
    "E": "λabcde.ab(cde)",
    "Ê": "λabcdefg.a(bcd)(efg)",
    "F": "λabc.cba",
    "F*": "λabcd.adcb",
    "F**": "λabcde.abedc",
    "G": "λabcd.ad(bc)",
    "H": "λabc.abcb",
    "I": "λa.a",
    "I*": "λab.ab",
    "I**": "λabc.abc",
    "ɩ": "λa.aSK",
    "J": "λabcd.ab(adc)",
    "K": "λab.a",
    "L": "λab.a(bb)",
    "M": "λa.aa",
    "M₂": "λab.ab(ab)",
    "O": "λab.b(ab)",
    "π": "λab.b",
    "Φ": "λabcd.a(bd)(cd)",
    "Φ₁": "λabcde.a(bde)(cde)",
    "Ψ": "λabcd.a(bc)(bd)",
    "Q": "λabc.b(ac)",
    "Q₁": "λabc.a(cb)",
    "Q₂": "λabc.b(ca)",
    "Q₃": "λabc.c(ab)",
    "Q₄": "λabc.c(ba)",
    "R": "λabc.bca",
    "R*": "λabcd.acdb",
    "R**": "λabcde.abdec",
    "S": "λabc.ac(bc)",
    "T": "λab.ba",
    "U": "λab.b(aab)",
    "V": "λabc.cab",
    "V*": "λabcd.adbc",
    "V**": "λabcde.abecd",
    "W": "λab.abb",
    "W¹": "λab.baa",
    "W*": "λabc.abcc",
    "W**": "λabcd.abcdd",
}


def remove_braces(s: str) -> str:
    """Return argument without enclosing parenthesis or the string itself."""
    return s[1:-1] if s and s[0] == "(" and s[-1] == ")" else s


class Obj:
    """Base class for node in the graph representation of a lambda expression."""

    def is_free_in_context(self, _v: Var) -> bool:
        """Test whether this is an object for a free variable in the context.  This is the generic
        implementation."""
        return True

    @override
    def __str__(self) -> str:
        raise NotImplementedError("__str__ called for Obj")

    def fmt(self, _varmap: Naming, _highlight: bool) -> str:
        """Format an expression as a string.  This pure virtual version must never
        be called."""
        raise NotImplementedError("fmt called for Obj")

    def replace(self, _v: Var, _expr: Obj) -> Obj:
        """Return the expression with the given variable replaced by the expression."""
        return self

    def duplicate(self) -> Obj:
        """Duplicate the object."""
        return self

    def recombine(self) -> Obj:
        """Recombine combinators."""
        return self

    def rmatch(self, other: Obj, _var_map: dict[Var, Obj]) -> bool:
        """Determine whether the expression matches OTHER considering variable renaming in VAR_MAP."""
        return self == other

    def collect_free_vars(self) -> set[str]:
        """Return a set of all the variables that are free for the entire expression."""
        return set()

    def collect_free_exprs(self) -> list[Obj]:
        """Return a set of all the expressions that consists only of free variables."""
        return list()

    def get_apps(self) -> list[Obj]:
        """Return a list of all the applications in the expression."""
        return []

    def unify_names(self) -> None:
        """Unify matched variable names."""
        pass


@final
class Var(Obj):
    """Object to represent a variable in the lambda expression graph.  This implements
    the de Bruijn notation by representing each new variable with a unique number."""

    varcnt: ClassVar[int] = 1

    def __init__(self, freename: str | None = None):
        self.id = Var.varcnt
        self.freename = freename
        self.identical_to: Var | Application | None = None
        Var.varcnt += 1

    @override
    def is_free_in_context(self, v: Var) -> bool:
        """Test whether this is a free variable.  If we come here and this is the variable
        we are looking for it is indeed free in the context."""
        return self.id != v.id

    @override
    def __str__(self):
        return f"{{var {self.id} → {self.identical_to.id if isinstance(self.identical_to, Var) else 'app'}}}" if self.identical_to else f"{{var {self.id}}}"

    @override
    def fmt(self, varmap: Naming, highlight: bool) -> str:
        res = self.freename or varmap.get(self)
        return f"{COLORS['freevar' if self.freename else 'var']}{res}{COLORS['off']}" if highlight else res

    @override
    def replace(self, v: Var, expr: Obj) -> Obj:
        return expr.duplicate() if v.id == self.id else self

    @override
    def rmatch(self, other: Obj, var_map: dict[Var, Obj]) -> bool:
        if not isinstance(other, Var):
            return False
        if self.id == other.id or var_map.get(other) == self:
            return True
        if isinstance(var_map.get(other), Empty) and self not in var_map.values():
            var_map[other] = self
            return True
        return False

    @override
    def collect_free_vars(self) -> set[str]:
        return set(self.freename) if self.freename else set()

    @override
    def collect_free_exprs(self) -> list[Obj]:
        return [self] if self.freename else []


@final
class Empty(Obj):
    """Object returned to indicate errors when parsing lambda expressions and other situations when a non-existing
    object needs to be represented."""

    @override
    def __str__(self):
        return "{}"

    @override
    def fmt(self, varmap: Naming, highlight: bool) -> str:  # pylint: disable=unused-argument
        # This is a filler.  An object of this type should never really be used.
        return "○"


@final
class Constant(Obj):
    """Object to represent constants in the lambda expression graph."""

    def __init__(self, name: str):
        self.name = name

    @override
    def __str__(self):
        return f"{{const {self.name}}}"

    @override
    def fmt(self, _varmap: Naming, _highlight: bool) -> str:
        return f"{self.name} "


@final
class Combinator(Obj):
    """Object to represent a recombined combinator."""

    def __init__(self, combinator: str, arguments: list[Obj] | None = None):
        self.combinator = combinator
        self.arguments = [] if arguments is None else arguments

    @override
    def is_free_in_context(self, _v: Var) -> bool:
        return True

    @override
    def __str__(self):
        if self.arguments:
            return f"{{{self.combinator} {' '.join([str(a) for a in self.arguments])}}}"
        return self.combinator

    @override
    def fmt(self, varmap: Naming, highlight: bool) -> str:
        combres = f"{COLORS['combinator']}{self.combinator}{COLORS['off']}" if highlight else self.combinator
        if self.arguments:
            combres += " " + " ".join([a.fmt(varmap, highlight) for a in self.arguments])
        return combres

    @override
    def get_apps(self) -> list[Obj]:
        return list(itertools.chain.from_iterable(a.get_apps() for a in self.arguments))


@final
class Application(Obj):
    """Object to represent the application (call) to a function in the lambda
    expression graph."""

    def __init__(self, ls: list[Obj]):
        assert len(ls) >= 2
        self.code: list[Obj] = (ls[0].code + ls[1:]) if isinstance(ls[0], Application) else ls
        assert self.code
        self.identical_to: Var | Application | None = None

    @override
    def is_free_in_context(self, v: Var) -> bool:
        return all(e.is_free_in_context(v) for e in self.code)

    @override
    def __str__(self):
        return f"{{App {' '.join([str(a) for a in self.code])}}}"

    @override
    def fmt(self, varmap: Naming, highlight: bool) -> str:
        return f"({''.join([a.fmt(varmap, highlight) for a in self.code]).rstrip()})"

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
    def rmatch(self, other: Obj, var_map: dict[Var, Obj]) -> bool:
        if isinstance(other, Var) and other in var_map:
            other = var_map[other]
        return isinstance(other, Application) and len(self.code) == len(other.code) and all(a.rmatch(b, var_map) for a, b in zip(self.code, other.code))

    @override
    def collect_free_vars(self) -> set[str]:
        # It would be good to write this code in one line as per code below but basedpyright complains.
        #   return set().union(*[e.collect_free_vars() for e in self.code])
        res: set[str] = set()
        for e in self.code:
            res |= e.collect_free_vars()
        return res

    @override
    def collect_free_exprs(self) -> list[Obj]:
        res: list[Obj] = []
        all = True
        for e in self.code:
            rres = e.collect_free_exprs()
            all &= e in rres
            for r in rres:
                if r not in res:
                    res.append(r)
        if all:
            res.append(self)
        return res

    @override
    def get_apps(self) -> list[Obj]:
        return [self] + list(itertools.chain.from_iterable(a.get_apps() for a in self.code))

    @override
    def unify_names(self) -> None:
        for i, e in enumerate(self.code):
            if isinstance(e, Var):
                if e.identical_to and isinstance(e.identical_to, Var):
                    self.code[i] = e.identical_to
            else:
                e.unify_names()

    def beta(self) -> Obj:
        """Perform beta reduction on the given application.  This is called on a freshly
        created object but the reduction cannot be performed in the constructor because
        the result of the beta reduction can be something other than an application."""
        if not isinstance(self.code[0], Lambda):
            return self
        la = self.code[0]
        r = la.code.replace(la.params[0], self.code[1])
        if len(la.params) > 1:
            r = newlambda(la.params[1:], r)
        return apply([r] + self.code[2:])


@final
class Lambda(Obj):
    """Object to represent a lambda expression in the lambda expression graph."""

    def __init__(self, params: list[Var], code: Obj):
        if not params:
            raise SyntaxError("lambda parameter list cannot be empty")
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
        return f"{{lambda {' '.join([str(a) for a in self.params])}.{str(self.code)}}}"

    @override
    def fmt(self, varmap: Naming, highlight: bool) -> str:
        # It is important to process the params first to ensure correct naming of parameter variables.
        # Assign new names to the parameter variables
        nvarmap = Naming(varmap.avoid)
        paramstr = "".join([a.fmt(nvarmap, highlight) for a in self.params])
        varmap.add(nvarmap)
        la = f"{COLORS['lambda']}λ{COLORS['off']}" if highlight else "λ"
        return f"({la}{paramstr}.{remove_braces(self.code.fmt(varmap, highlight))})"

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
        free_exprs = self.collect_free_exprs()
        rself = Lambda(self.params, self.code.recombine())
        for comb, combstr in KNOWN_COMBINATORS.items():
            combexpr = from_string(combstr)
            assert isinstance(combexpr, Lambda)
            if len(self.params) == len(combexpr.params):
                if self.rmatch(combexpr, {}):
                    return Combinator(
                        comb,
                        [],
                    )
            elif len(self.params) < len(combexpr.params):
                # Recognize simplification like those by Augustsson
                #   S (K a) (K b) -> λc.ab -> K (a b)
                #   S (K a) b -> λc.a(bc) -> B a b
                # I.e., we create a combinator use where free variables are used as constant parameters
                for fixed_params in itertools.product(free_exprs, repeat=len(combexpr.params) - len(self.params)):
                    param_map = {p[0]: p[1] for p in zip(combexpr.params, fixed_params)}
                    if self.rmatch(combexpr, param_map):
                        return Combinator(comb, list(fixed_params))
        return rself

    @override
    def rmatch(self, other: Obj, var_map: dict[Var, Obj]) -> bool:
        if not isinstance(other, Lambda):
            return False
        newvar_map = var_map.copy()
        # We reverse the parameters in case the self lambda has fewer parameters which have been filled
        # with expressions consisting of free variables in recombine.
        for param, newparam in zip(reversed(self.params), reversed(other.params)):
            assert newparam not in newvar_map
            assert param not in newvar_map
            newvar_map[newparam] = param
        assert all(p in newvar_map for p in other.params)
        if self.code.rmatch(other.code, newvar_map):
            # Propagate the newly found values back to the caller.
            for k in var_map:
                if isinstance(var_map[k], Empty):
                    var_map[k] = newvar_map[k]
                else:
                    assert var_map[k] == newvar_map[k]
            return True
        return False

    @override
    def collect_free_vars(self) -> set[str]:
        return self.code.collect_free_vars()

    @override
    def collect_free_exprs(self) -> list[Obj]:
        return self.code.collect_free_exprs()

    @override
    def get_apps(self) -> list[Obj]:
        return self.code.get_apps()

    @override
    def unify_names(self) -> None:
        for i, e in enumerate(self.params):
            if e.identical_to and isinstance(e.identical_to, Var):
                self.params[i] = e.identical_to
        self.code.unify_names()


def parse_lambda(s: str, ctx: dict[str, Var]) -> tuple[Obj, str]:
    """Parse the representation of a lambda definition.  Return the graph
    representation and the remainder of the string not part of the just
    parsed part."""
    assert s[0] == "λ"
    s = s[1:].strip()
    params: list[Var] = []
    recctx = ctx.copy()
    while s:
        if s[0] == ".":
            break
        if s[0] not in VARIABLE_NAMES:
            raise SyntaxError(f"invalid λ parameters {s[0]}")
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
    body: list[Obj] = []
    while s:
        if s[0].isspace():
            break
        e, s = parse_one(s, recctx)
        body.append(e)
    return newlambda(params, apply(body)), s


def parse_paren(s: str, ctx: dict[str, Var]) -> tuple[Obj, str]:
    """Parse an expression in parenthesis.  Return the graph
    representation and the remainder of the string not part of the just
    parsed part."""
    assert s[0] == "("
    start = 1
    end = 1
    depth = 0
    while True:
        if end == len(s):
            raise SyntaxError("incomplete parenthesis")
        if s[end] == ")":
            if depth == 0:
                res = parse_top(s[start:end], ctx)
                return res, s[end + 1 :]
            depth -= 1
        if s[end] == "(":
            depth += 1
        end += 1


def get_constant(s: str) -> tuple[Obj, str]:
    """Parse a constant name.  Unlike a variable, a constant can be longer than a
    single character and therefore has to be terminated by either of the special
    characters using the string representation (λ, (, ), or .) or a whitespace.
    Return the graph representation and the remainder of the string not part of
    the just parsed part."""
    delimiters = {")", "(", ".", "λ", "\n", "\t", "\v", "\f", " ", "\r"}
    end_idx = 0
    while end_idx < len(s) and s[end_idx] not in delimiters:
        end_idx += 1

    token = s[:end_idx]
    # Find the longest known combinator that is a prefix of the token.
    for i in range(len(token), 0, -1):
        # If the token at this position would have one of the known suffixes it must include the latter.
        if i < end_idx and token[i] in {"*", "₁", "₂", "₃", "₄", "¹", "#"}:
            continue
        prefix = token[:i]
        if prefix in KNOWN_COMBINATORS:
            expr = parse_top(KNOWN_COMBINATORS[prefix], {})
            return expr, s[i:].lstrip()

    # No known combinator found, treat the whole token as a constant.
    return Constant(token), s[end_idx:].lstrip()


def parse_one(s: str, ctx: dict[str, Var]) -> tuple[Obj, str]:
    """Toplevel function to parse a string representation of a lambda expression.
    Return the graph representation and the remainder of the string not part of
    the just parsed part."""
    match s[0]:
        case "λ":
            return parse_lambda(s, ctx)
        case "(":
            return parse_paren(s, ctx)
        case c if c in VARIABLE_NAMES:
            return ctx[s[0]] if s[0] in ctx else Var(s[0]), s[1:]
        case c if c.isalpha():
            return get_constant(s)
        case _:
            raise SyntaxError(f"cannot parse {s}")


def newlambda(params: list[Var], code: Obj) -> Obj:
    """Create a new lambda expression using the given parametesr and body of code.
    But the function also performs η-reduction, i.e., it returns just the function
    expression (first of the application values) in case the resulting lambda would
    just apply the required parameter(s) to the application value in order."""
    if isinstance(code, Application) and len(params) < len(code.code):
        ncode = len(code.code) - len(params)
        if params == code.code[ncode:] and all(c.is_free_in_context(e) for e in params for c in code.code[:ncode]):
            return code.code[0] if ncode == 1 else Application(code.code[:ncode])
    return Lambda(params, code)


def apply(li: list[Obj]) -> Obj:
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


def parse_top(s: str, ctx: dict[str, Var]) -> Obj:
    """Parse a string to a lambda expression, taking one part at a time
    an creating an application expression from the parts.  Return the graph
    representation and the remainder of the string not part of the just
    parsed part."""
    s = s.strip()
    res: list[Obj] = []
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


class Vargen:  # pylint: disable=too-few-public-methods
    """Class to generate unique, consecutive variable names."""

    def __init__(self):
        self.typeidx: int = 0

    def name(self):
        "Generate the name."
        res = TYPE_NAMES[self.typeidx]
        self.typeidx += 1
        return res


class Type:
    """Simple type object used in type signatures."""

    def __init__(self):
        self.name: str | None = None

    @override
    def __repr__(self):
        return f"Type({self.name})"

    @override
    def __str__(self):
        assert self.name is not None
        return self.name

    def finalize(self, gen: Vargen) -> Self:
        """Create variable name."""
        if self.name is None:
            self.name = gen.name()
        return self


@final
class TypeFunc(Type):
    """Type object for functions."""

    def __init__(self, app: Application):
        super().__init__()
        self.args = app
        self.ret = None

    @override
    def __repr__(self):
        return f"TypeFunc({self.args})"


def determine_types(obj: Obj, types: dict[Obj, list[Type]]) -> dict[Obj, list[Type]]:
    """Traverse the expression object and extract type information and handles for variables and
    function invocations."""
    match obj:
        case Var():
            if obj not in types:
                types[obj] = [Type()]
        case Application():
            assert obj not in types
            for a in obj.code:
                types = determine_types(a, types)
            tf = TypeFunc(obj)
            if obj.code[0] in types:
                # This can only happen for variables
                assert isinstance(obj.code[0], Var)
                if not isinstance(types[obj.code[0]][0], TypeFunc):
                    types[obj.code[0]] = [tf]
                else:
                    types[obj.code[0]].append(tf)
            else:
                types[obj.code[0]] = [tf]
            types[obj] = [Type()]
        case Lambda():
            types = determine_types(obj.code, {})
            for p in obj.params:
                if p not in types:
                    types[p] = [Type()]
        case Combinator():
            raise RuntimeError("Combinator cannot be handled in collect_types")
        case _:
            raise NotImplementedError(f"Unexpected type #1 {type(obj)}")
    return types


def type_resolve(obj: Var | Application) -> Var | Application:
    """Determine the actual, non-aliased type of the object."""
    while obj.identical_to:
        obj = obj.identical_to
    return obj


def notation(obj: Obj, types: dict[Obj, Type], gen: Vargen) -> str:
    """Create a Haskell-like notation for the signature of the expression."""
    match obj:
        case Var():
            assert obj in types
            if obj.identical_to:
                return notation(obj.identical_to, types, gen)
            return str(types[obj].finalize(gen))
        case Lambda():
            res: list[str] = []
            if isinstance(obj.code, Var):
                for p in obj.params:
                    res.append(notation(p, types, gen))
            else:
                assert isinstance(obj.code, Application)
                for p in obj.params:
                    assert p in types
                    match types[p]:
                        case TypeFunc():
                            args = cast(TypeFunc, types[p]).args
                            s = "("
                            for c in args.code[1:]:
                                s += f"{notation(c, types, gen)} → "
                            ttype = types[type_resolve(args)].finalize(gen)
                            res.append(f"{s}{str(ttype)})")
                        case Type():
                            rp = type_resolve(p)
                            res.append(f"{str(types[rp].finalize(gen))}")
                assert isinstance(types[obj.code], Type)
            res.append(notation(obj.code, types, gen))
            return " → ".join(res)
        case Application():
            assert obj in types
            if obj.identical_to:
                return notation(obj.identical_to, types, gen)
            return str(types[obj].finalize(gen))
        case _:
            raise NotImplementedError(f"Unexpected type #2 {type(obj)}")


def mark_identical(left: Obj, right: Obj):
    """If two objects are recognized after their creation to have the same type, mark them as identical."""
    if not isinstance(left, Var) and not isinstance(left, Application):
        raise NotImplementedError(f"Unexpected types #3 {type(left)}")
    if not isinstance(right, Var) and not isinstance(right, Application):
        raise NotImplementedError(f"Unexpected types #3a {type(right)}")
    if isinstance(left, Var):
        right.identical_to = left
    elif isinstance(right, Var):
        left.identical_to = right
    else:
        assert isinstance(left, Application) and isinstance(right, Application)
        left.identical_to = right


def to_typesig(expr: Obj, _highlight: bool = False) -> str:
    """Return a string representation for type signature of the expression."""
    gen = Vargen()
    if not isinstance(expr, Lambda):
        assert isinstance(expr, (Var, Application))
        t = Type()
        return str(t.finalize(gen))

    # types is a dictionary associating the expression and subexpression objects with deduced
    # types.  E.g., for
    #    λabc.ab
    # the dictionary contains a
    #   {Var(a): [TypeFunc({App {Var(a)} {Var(b)}})], Var(b): [Type(None)], Var(c): [Type(None)], {App {Var(a)} {Var(b)}}: [Type(None)]}
    # Type(None) indicates that no type is assigned yet.
    types = determine_types(expr, {})
    stypes: dict[Obj, Type] = {}
    rtypes: dict[Obj, Type] = {}
    for k, t in types.items():
        if len(t) != 1:
            assert all(isinstance(tt, TypeFunc) for tt in t)
            for tt in t[1:]:
                # We cannot modify types while iterating over it and we cannot add the corrected
                # value to stypes because the iteration order might cause the value to be overwritten.
                rtypes[cast(TypeFunc, tt).args] = types[cast(TypeFunc, t[0]).args][0]
                assert cast(TypeFunc, t[0]).args.code[0] == cast(TypeFunc, tt).args.code[0]
                if len(cast(TypeFunc, t[0]).args.code) != len(cast(TypeFunc, tt).args.code):
                    raise NotImplementedError("unequal number of parameters of matching invocations")
                for e in zip(cast(TypeFunc, t[0]).args.code[1:], cast(TypeFunc, tt).args.code[1:]):
                    mark_identical(e[0], e[1])
                mark_identical(cast(TypeFunc, tt).args, cast(TypeFunc, t[0]).args)
        stypes[k] = t[0]

    # Give preference to the types determined through the function call matching
    for k, v in rtypes.items():
        stypes[k] = v
    # Just to make sure it is known that types and rtypes are not used after this point
    del types
    del rtypes

    expr.unify_names()

    return notation(expr, stypes, gen)


def decode_number(expr: Obj) -> str:
    """Decode Church encoding for numbers."""
    res = ""
    if isinstance(expr, Lambda):
        la = cast(Lambda, expr)
        if len(la.params) == 2:
            c = la.code
            n = 0
            while c != la.params[1]:
                if not isinstance(c, Application):
                    return ""
                a = cast(Application, c)
                if len(a.code) != 2 or a.code[0] != la.params[0]:
                    return ""
                c = a.code[1]
                n += 1
            res = str(n)
    return res


def handle(a: str, church: bool, echo: bool) -> int:
    """Parse given string, simplify, and print the lambda expression."""
    ec = 0
    input_prompt = f"{COLORS['input_prompt']}»{COLORS['off']} "
    output_prompt = f"{COLORS['output_prompt']}⇒{COLORS['off']} " if is_terminal else ""
    typesig_prompt = f"{COLORS['output_prompt']}🖊{COLORS['off']} " if is_terminal else ""
    separator_len = os.get_terminal_size()[0] if is_terminal else 72

    if echo and is_terminal:
        print(f"{input_prompt}{a}")
    try:
        expr = from_string(a)
        number = ""
        if church:
            number = decode_number(expr)
            if number:
                number = f" = {number}"
        print(f"{output_prompt}{to_string(expr, is_terminal)}{number}")
        print(f"{typesig_prompt}{to_typesig(expr, is_terminal)}")
    except SyntaxError as e:
        print(f'eval("{a}") failed: {e.args[0]}')
        ec = 1
    if not echo:
        print("\u2501" * separator_len)
    return ec


def repl(church: bool) -> int:
    """This is the REPL."""
    ec = 0
    try:
        input_prefix = f"{COLORS['input_prompt']}»{COLORS['off']} "
        while True:
            s = input(input_prefix)
            if not s:
                break
            ec = ec | handle(s, church, False)
    except EOFError:
        print("")
    return ec


def check() -> int:
    """Sanity checks.  Return error code that is used as the exit code of the process."""
    combinator_checks = [
        ("S K K", "I"),
        ("K I", "π"),
        ("K (S K K)", "π"),
        ("B B", "D"),
        ("B D", "D₁"),
        ("B (B B)", "D₁"),
        ("D D", "D₂"),
        ("B B D", "D₂"),
        ("D (B B)", "D₂"),
        ("B B (B B)", "D₂"),
        ("B B₁", "E"),
        ("B (D B)", "E"),
        ("B (B B B)", "E"),
        ("D B", "B₁"),
        ("D B₁", "B₂"),
        ("B D B", "B₃"),
        ("S(B B S)(K K)", "C"),
        ("C I", "T"),
        ("S(K S)K", "B"),
        ("B₁ S B", "Φ"),
        ("B Φ Φ", "Φ₁"),
        ("B (Φ B S) K K", "C"),
        ("B(S Φ C B)B", "Ψ"),
        ("λx.NotX x", "NotX"),
        ("B B C", "G"),
        ("E T T E T", "F"),
        ("B W (B C)", "H"),
        ("B(B C)(W(B C(B(B B B))))", "J"),
        ("C B M", "L"),
        ("B M", "M₂"),
        ("S I", "O"),
        ("C B", "Q"),
        ("B C B", "Q₁"),
        ("C(B C B)", "Q₂"),
        ("B T", "Q₃"),
        ("F* B", "Q₄"),
        ("B B T", "R"),
        ("L O", "U"),
        ("B C T", "V"),
        ("C(B M R)", "W"),
        ("C W", "W¹"),
        ("S(S K)", "I*"),
        ("B W", "W*"),
        ("B C", "C*"),
        ("B(B W)", "W**"),
        ("B C*", "C**"),
        ("C* C*", "R*"),
        ("B(B B B)(B(B B B))", "Ê"),
        ("λabcd.MMMabcd", "MMM"),
        ("λabcd.MMMabdc", "λabcd.MMMabdc"),
        ("λabcd.MMMabc", "λabcd.MMMabc"),
        ("S(K e)I", "e"),
        ("B C* R*", "F*"),
        ("C* F*", "V*"),
        ("B C*", "C**"),
        ("B R*", "R**"),
        ("B F*", "F**"),
        ("B V*", "V**"),
        ("S(S(K(S(K S)K))S)(K K)", "C"),
        ("S(K(S(S(K(S(K S)K))S)(K K)))", "C*"),
        ("C(B(B(B W)(B B C))(B(B C)(B B(B B))))(B W K)", "Ψ"),
        ("S(K(S(S(K S)K)))K", "Q"),
        ("S(K(S(K S)K))(S(K(S(S K K)))K)", "R"),
        ("B B(C(W K))", "R"),
        ("S(K(S(S(K(S(K S)K))S)(K K)))(S(K(S(S(K(S(K S)K))S)(K K))))", "R*"),
        ("B C(B C)", "R*"),
        ("B(B W)(B B C)", "S"),
        ("B(B(B(B W)(B B C)))B", "Φ"),
        ("C(W K)", "T"),
        ("S(K(S(S K K)))K", "T"),
        ("S S(S K)", "W"),
        # Universal Iota
        ("ɩɩ", "I"),
        ("ɩ(ɩ(ɩɩ))", "K"),
        ("ɩ(ɩ(ɩ(ɩɩ)))", "S"),
    ]
    simplification_checks: list[tuple[str, str]] = [
        # Simplification rules (Augustsson)
        ("S (K a) I", "a"),
        ("S (K a) (K b)", "K (ab)"),
        ("S (K a) b", "B a b"),
        ("S a (K b)", "C a b"),
        ("S (B a b) c", "Φ a b c"),
        ("C (B a b) c", "C# a b c"),
        ("B (a b) c", "D a b c"),
    ]
    ec = 0
    print("Combinator checks")
    all_tests = [(key, key) for key in KNOWN_COMBINATORS] + combinator_checks + simplification_checks
    for testinput, expected in all_tests:
        resexpr = from_string(testinput)
        res = to_string(resexpr)
        if res != expected:
            if expected in KNOWN_COMBINATORS:
                print(f"❌ {testinput} ⇒ {res} {resexpr} but {expected} {from_string(expected)} = {KNOWN_COMBINATORS[expected]} expected")
            else:
                print(f"❌ {testinput} ⇒ {res} {resexpr} but {expected} {from_string(expected)} expected")
            ec = 1
        else:
            print(f"✅ {testinput} ⇒ {res}")

    signature_checks = [
        ("B", "(A → B) → (C → A) → C → B"),
        ("B₁", "(A → B) → (C → D → A) → C → D → B"),
        ("B₂", "(A → B) → (C → D → E → A) → C → D → E → B"),
        ("B₃", "(A → B) → (C → A) → (D → C) → D → B"),
        ("C", "(A → B → C) → B → A → C"),
        ("C*", "(A → B → C → D) → A → C → B → D"),
        ("C**", "(A → B → C → D → E) → A → B → D → C → E"),
        ("C#", "(A → B → C) → (D → A) → B → D → C"),
        ("D", "(A → B → C) → A → (D → B) → D → C"),
        ("D₁", "(A → B → C → D) → A → B → (E → C) → E → D"),
        ("D₂", "(A → B → C) → (D → A) → D → (E → B) → E → C"),
        ("E", "(A → B → C) → A → (D → E → B) → D → E → C"),
        ("Ê", "(A → B → C) → (D → E → A) → D → E → (F → G → B) → F → G → C"),
        ("F", "A → B → (B → A → C) → C"),
        ("F*", "(A → B → C → D) → C → B → A → D"),
        ("F**", "(A → B → C → D → E) → A → D → C → B → E"),
        ("G", "(A → B → C) → (D → B) → D → A → C"),
        ("H", "(A → B → A → C) → A → B → C"),
        ("I", "A → A"),
        ("I*", "(A → B) → A → B"),
        ("I**", "(A → B → C) → A → B → C"),
        # ɩ
        ("J", "(A → B → B) → A → B → A → B"),
        ("K", "A → B → A"),
        # L
        # M
        # M₂
        # O
        ("π", "A → B → B"),
        ("Φ", "(A → B → C) → (D → A) → (D → B) → D → C"),
        ("Φ₁", "(A → B → C) → (D → E → A) → (D → E → B) → D → E → C"),
        ("Ψ", "(A → A → B) → (C → A) → C → C → B"),
        ("Q", "(A → B) → (B → C) → A → C"),
        ("Q₁", "(A → B) → C → (C → A) → B"),
        ("Q₂", "A → (B → C) → (A → B) → C"),
        ("Q₃", "(A → B) → A → (B → C) → C"),
        ("Q₄", "A → (A → B) → (B → C) → C"),
        ("R", "A → (B → A → C) → B → C"),
        ("R*", "(A → B → C → D) → C → A → B → D"),
        ("R**", "(A → B → C → D → E) → A → D → B → C → E"),
        ("S", "(A → B → C) → (A → B) → A → C"),
        ("T", "A → (A → B) → B"),
        # U
        ("V", "A → B → (A → B → C) → C"),
        ("V*", "(A → B → C → D) → B → C → A → D"),
        ("V**", "(A → B → C → D → E) → A → C → D → B → E"),
        ("W", "(A → A → B) → A → B"),
        ("W¹", "A → (A → A → B) → B"),
        ("W*", "(A → B → B → C) → A → B → C"),
        ("W**", "(A → B → C → C → D) → A → B → C → D"),
        ("BK", "(A → B) → A → C → B"),
    ]
    print("\nSignature checks")
    for testinput, expected in signature_checks:
        resexpr = from_string(testinput)
        res = to_typesig(resexpr)
        if res != expected:
            print(f"❌ {testinput} ⇒ {res} but {expected} expected")
            ec = 1
        else:
            print(f"✅ {testinput} ⇒ {res}")
    return ec


def main() -> None:
    """Called as main function of the program."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--check", dest="check", action="store_true")
    _ = parser.add_argument("--church", dest="church", action="store_true")
    _ = parser.add_argument("expression", metavar="expression", type=str, nargs="*")
    args = parser.parse_args()

    if cast(bool, args.check):
        # Overwrite eventual user setting
        args.tracing = False
        ec = check()
    else:
        init_terminal()
        if cast(str, args.expression):
            ec = handle(" ".join(cast(str, args.expression)), args.church, True)
        else:
            ec = repl(args.church)
    sys.exit(ec)


if __name__ == "__main__":
    main()
