[![Build Status](https://github.com/drepper/combinatorial/workflows/CI/badge.svg)](https://github.com/drepper/combinatorial/actions)
[![License: MIT](license-MIT-blue.svg)](https://github.com/drepper/combinatorial/blob/master/LICENSE)

# Combinatorial Evaluator

This is a Python package for evaluating combinatorial expressions.  The parser understands a lambda notation and also
knows combinators such as the identity combinator (I), the constant combinator (K), the pair combinator (Ψ), and the
successor combinator (S).


## Usage

The code can be used in three ways:

1. to evaluate an expressions:

```bash
python3 combinatorial.py BBK
```

   Note that the output looks different depending on whether the output goes to a terminal or not.  In the latter case
   only the result is printed which means the tool can be used as a kind of calculator.

2. as a REPL when the Python code is run without an expression

3. as a module:

```Python
import combinator
e=combinator.from_string('BBK')
combinator.to_string(e)
```

## Functionality

The `from_string` function takes a string representation of a combinatorial expression and returns a corresponding
expression object.  While the expression is parsed beta and eta reductions are performed and, once completed, the
resulting expression is examined whether it can be better represented as a (series of) combinator(s).

The type of the return value is an overloaded class named `Obj`.  The actual implementation might change, one should
not depend on any details of the implementation.

The `Obj` object can be passed to the `to_string` function to obtain a string representation of the expression.
For expressions which did not undergo any reduction, the `to_string` function returns the original string representation,
unless a concise combinator representation is used.


## Currently Supported Combinators

The notation used for the combinators were mostly derived from the table on [Chris Rathman's site](https://www.angelfire.com/tx4/cus/combinator/birds.html).  The actual names should not matter and anyone
could replace them with their own.  The only dependency on the names in the implementation is that suffixes,
such as ₁ or *, are hardcoded in the scanner code.  But the list of suffixes is also easy to change or extend.

|Name|Lambda Expression|
|:-|:-|
|B| λabc.a(bc)|
|B₁| λabcd.a(bcd)|
|B₂| λabcde.a(bcde)|
|B₃| λabcd.a(b(cd))|
|C| λabc.acb|
|C*| λabcd.abdc|
|C**| λabcde.abced|
|D| λabcd.ab(cd)|
|D₁| λabcde.abc(de)|
|D₂| λabcde.a(bc)(de)|
|E| λabcde.ab(cde)|
|Ê| λabcdefg.a(bcd)(efg)|
|F| λabc.cba|
|F*| λabcd.adcb|
|F**| λabcde.abedc|
|G| λabcd.ad(bc)|
|H| λabc.abcb|
|I| λa.a|
|I*| λab.ab|
|I**| λabc.abc|
|J| λabcd.ab(adc)|
|K| λab.a|
|L| λab.a(bb)|
|M| λa.aa|
|M₂| λab.ab(ab)|
|O| λab.b(ab)|
|π| λab.b|
|Φ| λabcd.a(bd)(cd)|
|Φ₁| λabcde.a(bde)(cde)|
|Ψ| λabcd.a(bc)(bd)|
|Q| λabc.b(ac)|
|Q₁| λabc.a(cb)|
|Q₂| λabc.b(ca)|
|Q₃| λabc.c(ab)|
|Q₄| λabc.c(ba)|
|R| λabc.bca|
|R*| λabcd.acdb|
|R**| λabcde.abdec|
|S| λabc.ac(bc)|
|T| λab.ba|
|U| λab.b(aab)|
|V| λabc.cab|
|V*| λabcd.adbc|
|V**| λabcde.abecd|
|W| λab.abb|
|W¹| λab.baa|
|W*| λabc.abcc|
|W**| λabcd.abcdd|

Since this list also includes the `S` and `K` combinators one can use the code to verify the reductions:

```
$ python3 combinator.py '((S(K((S((SK)K))(K((S(K(S((SK)K))))K)))))((S(K((S(K((S(KS))K)))((S(KS))K))))((S(K(S((SK)K))))K)))'
» ((S(K((S((SK)K))(K((S(K(S((SK)K))))K)))))((S(K((S(K((S(KS))K)))((S(KS))K))))((S(K(S((SK)K))))K)))
⇒ F
```


## Limitations

The handling of the `M` combinator (Mockingbird) and related once is *anything* but good.  The recursive nature needs
to be handled differently.  At least the current implementation does not spiral out of control, most of the time.

Additionally, the scanners requires that variables are represented by lowercase letters.  This is also and arbitrary choice
but seems to be commonly used.  This limits the number of variables to 26, of cource.  It is easy enough to extend, just
modify the `VARIABLE_NAMES` string in the sources.  There should be no overlap with the characters used in combinators.

Note that free variables limit the number of variables used throughout the expression further.  The names must be unique
and hence are reserved.  Otherwise, scoping rules of lambda calculus specifies how to interpret names of parameters which
also appear in outer scopes.
