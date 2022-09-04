---
layout: post
mathjax: false
title:  "Partitioning a value range with a binary tree"
date:   2022-09-04
---

**Code for this post** is available at: (https://github.com/agoryuno/offcuts/blob/main/blog/binary_range.py)

We want to subdivide a range of numeric values into subranges in a way not unsimilar to [binary space partitioning](https://en.wikipedia.org/wiki/Binary_space_partitioning). The initial range is divided into two more or less equal subranges, each of those is divided into two more and so on. The subdivision stops whenever it reaches a given minimal range size. What we end up with is a binary tree with ranges on the nodes.

The practical purpose of this exercise is to optimize queries to a certain API, which returns a number of search results given the minimum and maximum values of a range as parameters. We want to select parameter pairs in such a way that when executing the actual search queries we get a maximum of N results per query, exceeding N for some queries only if the parameters represent a minimal range size.

To achieve this end we'll first build a binary tree of search parameters and then traverse it, abandoning a branch every time we get the number of results less than or equal to N. If we get to the end of a branch and still get more than N results - we'll just live with it.

## The nodes

The first item we need to deal with is the tree node. The dataclass below represents a node of a tree. It has two fields for the bottom and top values of a range and two fields for the left and right branch of this node.

We'll also add an iterator to the class (the `__iter__` method) which will recursively visit each node from left to right, returning each node object on the tree. 

To be able to compare nodes to each other we define the `_range()` method. This returns the size of the range represented by the `minvalue` and `maxvalue` fields as the metric of the node so that the larger range corresponds to the larger node.


```python
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Union

# We don't include complex numbers
# in our Numeric data type for simiplicity's
# sake
Numeric = Union[int, float]

@dataclass
class ValueRange:
    minvalue: Numeric
    maxvalue: Numeric
        
    left : ValueRange = None
    right : ValueRange = None
        
    def __iter__(self):
        if self.left is not None:
            for elem in self.left:
                yield elem
        yield self
        if self.right is not None:
            for elem in self.right:
                yield elem
                
    def _range(self):
        return self.maxvalue - self.minvalue
    
    def __eq__(self, other):
        return self._range() == other._range()
    
    def __lt__(self, other):
        return self._range() < other._range()
    
    def __le__(self, other):
        return self._range() <= other._range()
    
    def __gt__(self, other):
        return self._range() > other._range()
    
    def __ge__(self, other):
        return self._range() >= other._range()
```

One technical note here is that importing `annotations` from `__future__` allows us to reference the `ValueRange` class in the type annotations of the class's fields. Without it we'd get a `NameError` exception, telling us that we are trying to use `ValueRange` before it's been fully declared.

## Building the tree

In order to build the actual tree we use a recursive function below. This takes a node and a minimum range size as parameters, calculates the midrange point and links the nodes together into a binary tree.

We are more concerned with the midpoint values of ranges being "pretty" than with them being actual exact midpoints. The API likes round numbers and can give erroneous results if presented with overly precise floating point parameter values. So we use the trick below to calculate midpoints in terms of `floor_range` *numeraires*. As long as the `floor_range` value is nice and round this method will give us nice and round midpoint values, at the cost of somewhat unequal subranges.


```python
def build_tree(val_range, floor_range):
    minval, maxval = val_range.minvalue, val_range.maxvalue
    
    floors = ((maxval - minval) // 2) / floor_range
    if floors < 1:
        return val_range
    
    midval = minval + math.floor(floors)*floor_range
    val_range.left = build_tree(ValueRange(minval, midval), floor_range)
    val_range.right = build_tree(ValueRange(midval, maxval), floor_range)
    
    return val_range
```

## Testing

Checking to see if the algorithm will be practical on household hardware, we'll test it on the range of 0 to 10 billion with a minimum range of 100 000.


```python
%timeit build_tree(ValueRange(0, 10e+9), 100e+3)
```

The mean timing on an old Core i5 quad core mobile CPU is 273ms per run, which is perfectly sufficient.

We might as well test the sorting speed to get an idea of how long the sorted tree takes to traverse:


```python
res = build_tree(ValueRange(0, 10e+9), 100e+3)
%timeit sorted([elem for elem in res], reverse=True)
```

    752 ms ± 31.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


This takes around 750ms on the same hardware. So the creation and sorting of a tree with that range of values would take about a second on a fairly outdated PC.
