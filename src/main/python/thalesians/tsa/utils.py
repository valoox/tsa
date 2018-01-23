"""Various miscellaneous utilites"""

import collections as col
import copy
import itertools
import math
import operator

from thalesians.tsa.compat import irange, abstract, abstractmethod
import thalesians.tsa.intervals as intervals


def cmp(x, y):
    """Int comparator of 'x' and 'y', i.e. cmp(x, y):
     * -1 if x  < y
     *  0 if x == y
     * +1 if x >  y
    """
    return (x > y) - (x < y)


def most_common(iterable):
    """Returns the most common element in a collection in a single pass

    Parameters
    ----------
    iterable: Iterable[T]
        The collection; note that this is being walked only once, so suitable
        for a single-pass iterable (e.g. a generator)

    Returns
    -------
    elt: T
        The most common element in the collection
    """
    sorted_iterable = sorted((x, i) for i, x in enumerate(iterable))
    groups = itertools.groupby(sorted_iterable, key=operator.itemgetter(0))

    def _auxfun(g):
        _, it = g
        count = 0
        min_index = len(sorted_iterable)
        for _, where in it:
            count += 1
            min_index = min(min_index, where)
        return count, -min_index
    return max(groups, key=_auxfun)[0]


def prepend(collection, to_prepend, in_place=False):
    """Generic prepender to a collection which supports item assignment

    Parameters
    ----------
    collection: list-like collection
        The collection (typically a list)
    to_prepend: collection
        The items to prepend to the collection
    in_place: bool, optional [default=False]
        Whether to modify the existing collection by side effect, or make a
        full copy of it and prepend it there

    Returns
    -------
    collection: list-like collection
        The collection, with item prepended
    """
    if not in_place:
        collection = copy.copy(collection)
    collection[0:0] = to_prepend
    return collection


def _pad_on_left_with_callable(collection, new_len, padding=None):
    """Pads the provided collection on the left by repeatedly calling the
    provided 'padding' callable, until 'new_len' is reached

    Parameters
    ----------
    collection: list-like collection
        The collection being padded
    new_len: int
        The target length
    padding: callable() -> Any, optional
        The padding callable; default simply fills it with 'None'

    Returns
    -------
    collection: list-like collection
        The original collection, left-padded with results of the callable
    """
    if padding is None:
        padding = lambda: None
    return prepend(
        collection,
        (padding() for _ in range(new_len - len(collection))),
        in_place=True
    )


def _pad_on_left_with_noncallable(collection, new_len, padding=None):
    """Pads the collection until it reached the target length

    Parameters
    ----------
    collection: list-like collection
        The collection to be padded
    new_len: int
        The target length of the collection
    padding: Any, optional [default=None]
        The padding element

    Returns
    -------
    collection: list-like collection
        The collection, left-padded with the provided element
    """
    return prepend(
        collection,
        (padding for _ in range(new_len - len(collection))),
        in_place=True
    )


def pad_on_left(collection, new_len, padding=None, in_place=False):
    """Pads the provided collection on the left

    Parameters
    ----------
    collection: list-like collection
        The collection of elements to be padded
    new_len: int
        The target length of the collection
    padding: Any | callable() -> Any
        The padding element/procedure
    in_place: bool, optional [default=False]
        Whether to modify the collection in place by side effect or make a
        (shallow) copy of it (default)

    Returns
    -------
    padded: list-like collection
        The padded collection
    """
    if not in_place:
        collection = copy.copy(collection)
    if callable(padding):
        return _pad_on_left_with_callable(collection, new_len, padding)
    else:
        return _pad_on_left_with_noncallable(collection, new_len, padding)


def _pad_on_right_with_callable(collection, new_len, padding=None):
    """Pads the provided collection on the right by repeatedly calling the
    provided 'padding' callable, until 'new_len' is reached

    Parameters
    ----------
    collection: list-like collection
        The collection being padded
    new_len: int
        The target length
    padding: callable() -> Any, optional
        The padding callable; default simply fills it with 'None'

    Returns
    -------
    collection: list-like collection
        The original collection, right-padded with results of the callable
    """
    collection.extend(padding() for _ in range(new_len - len(collection)))
    return collection


def _pad_on_right_with_noncallable(collection, new_len, padding=None):
    """Pads the collection until it reached the target length

    Parameters
    ----------
    collection: list-like collection
        The collection to be padded
    new_len: int
        The target length of the collection
    padding: Any, optional [default=None]
        The padding element

    Returns
    -------
    collection: list-like collection
        The collection, right-padded with the provided element
    """
    collection.extend(padding for _ in range(new_len - len(collection)))
    return collection


def pad_on_right(collection, new_len, padding=None, in_place=False):
    """Pads the provided collection on the right with the provided element(s)

    Parameters
    ----------
    collection: list-like collection
        The collection of elements to be padded
    new_len: int
        The target length of the collection
    padding: Any | callable() -> Any
        The padding element/procedure
    in_place: bool, optional [default=False]
        Whether to modify the collection in place by side effect or make a
        (shallow) copy of it (default)

    Returns
    -------
    padded: list-like collection
        The padded collection
    """
    if not in_place:
        collection = copy.copy(collection)
    if callable(padding):
        return _pad_on_right_with_callable(collection, new_len, padding)
    else:
        return _pad_on_right_with_noncallable(collection, new_len, padding)


def trim_on_left(collection, new_len, in_place=False):
    """Trims the collection to the provided length

    Parameters
    ----------
    collection: collection supporting slice deletion
        The collection being trimmed
    new_len: int
        The target length after trimming
    in_place: bool, optional [default=False]
        Whether to modify the collection in-place by side effect or returned
        a modified (shallow) copy (default)

    Returns
    -------
    trimmed: collection
        The collection, trimmed at the provided length
    """
    trim = max(len(collection) - new_len, 0)
    if not in_place:
        return collection[trim:]
    else:
        del collection[:trim]
        return collection


def trim_on_right(collection, new_len, in_place=False):
    """Trims the collection to the provided length

    Parameters
    ----------
    collection: collection supporting slice deletion
        The collection being trimmed
    new_len: int
        The target length after trimming
    in_place: bool, optional [default=False]
        Whether to modify the collection in-place by side effect or returned
        a modified (shallow) copy (default)

    Returns
    -------
    trimmed: collection
        The collection, trimmed at the provided length
    """
    if not in_place:
        return collection[:new_len]
    else:
        del collection[new_len:]
        return collection


def xconst(value):
    """A constant generator, yielding 'value' forever

    xconst(x) <=> (x for _ in itertools.count())

    Parameters
    ----------
    value: T
        The value being yielded

    Yields
    ------
    value: T
        The same value, forever
    """
    while True:
        yield value


def xbatch(size, iterable):
    """Batches the collection:
    xbatch(size, c) <=> (c[i:i+size] for i in range(0, len(c), size))

    Parameters
    ----------
    size: int
        The number of elements of each 'batch'
    iterable: collection
        The collection being split

    Yields
    ------
    batch: iterable[i:i+size]
        Subsets of the iterable of the specified batch size
    """
    l = len(iterable)
    for i in range(0, l, size):
        yield iterable[i:min(i + size, l)]


def batch(size, iterable):
    """Splits the collection into a number of 'batches' of length (at most) size

    batch(size, c) <=> [c[i:i+size] for i in range(0, len(c), size)]

    Parameters
    ----------
    size: int
        The maximum length of the batches
    iterable: collection
        The collection to be batched

    Returns
    -------
    batches: list[iterable[i:i+size]]
        All the distinct batches forming the original collection
    """
    return list(xbatch(size, iterable))


def peek(iterable, size=1):
    """Reads the first 'size' elements of the provided iterable, but doesn't
    pull them from the iterable

    Parameters
    ----------
    iterable: iterable, possibly infinite
        The collection of elements in which elements are peeked at
    size: int, optional [default=1]
        The number of elements to peek at

    Returns
    -------
    objs: list[Any](size)
        The first 'size' elements (or less, if iterable is empty)
    iterable: iterable
        The iterable, still containing objects in 'objs'

    Example
    -------
    >>> x = (x**2 for x in range(10))
    >>> f3, elts = peek(x, 3)
    >>> assert f3 == [1, 4, 9]
    >>> # Note that 'elts' still starts at '1'
    >>> assert next(elts) == 1

    The peeking stops when the generator does
    >>> x = (x for x in [1,2,3])
    >>> f5, elts = peek(x, 12)
    >>> # Although 12 elements were asked, only 3 are provided
    >>> assert len(f5) == 3

    This also works for infinite generators:
    >>> inf = (12 for _ in xconst(None))
    >>> x, inf = peek(inf)
    >>> assert x == 12
    """
    objs = [elt for _, elt in zip(irange(size), iterable)]
    return objs, itertools.chain(objs, iterable)


# A simple pair interval/offset
Bracket = col.namedtuple('Bracket', ('interval', 'offset'))


def bracket(
        iterable,
        origin,
        interval_size,
        already_sorted=False,
        intervals_right_closed=False,
        coalesce=False
):
    """This will generate a sequence of 'Bracket' objects and indices forming a
    covering of the provided collection of elements.
    The 'tiling' of the returned intervals will be regular (i.e. every interval
    will be of the form:
         [origin + n * interval_size; origin + (n+1) * interval_size)
    but intervals which do NOT contain any element will be skipped and not
    returned.

    Parameters
    ----------
    iterable: collection[T](n,)
        The collection of elements being covered by the brackets. Note that
        this is walked only once, and can be a generator
    origin: T
        The initial point from which the intervals should start
    interval_size: dT
        The requested size of the intervals
    already_sorted: bool, optional [default=False]
        If not (the default), this will sort the iterable first. If the
        collection is guaranteed to be sorted already, this might make it more
        efficient.
    intervals_right_closed: bool, optional [default=False]
        By default, this will generate a covering of left-closed, right-opened
        intervals, i.e. [a; b), [b; c), [c; d), ... , [y; z]
        Setting this to 'True' will instead have them closed on the right, i.e.
        [a; b], (b; c], (c; d], ..., (y; z]
    coalesce: bool, optional (default=False)
        If set to True, adjacent 'tiles' will be MERGED, i.e. a single Interval
        of size k * interval_size will be produced instead of 'k' distinct
        ones.

    Returns
    -------
    brackets: list[Bracket]
        The brackets forming the coverings
    indices: list[int](n,)
        For each element of the input, this contains the index of the bracket
        that this is a part of

    Examples
    --------
    >>> col = [1,2,3,4,10,12]
    >>> bra, idx = bracket(col, 0.5, 2.)
    >>> # Tiling will be [0.5; 2.5), [2.5; 4.5), [8.5; 10.5), [10.5, 12.5)
    >>> idx
    [0, 0, 1, 1, 2, 3]
    >>> print(bra[0].interval)
    [0.5; 2.5)
    >>> brc, idx = bracket(col, 0.5, 2., coalesce=True)
    >>> idx
    [0, 0, 0, 0, 1, 1]
    >>> print(brc[0].interval)
    [0.5; 4.5)
    >>> print(brc[1].interval)
    [8.5; 12.5)
    """
    if not already_sorted:
        sorted_indices, iterable = zip(*sorted(
            ((i, v) for i, v in enumerate(iterable)),
            key=operator.itemgetter(1)
        ))
    
    brackets = []
    bracket_indices = []

    # The current interval
    interval = None
    
    for x in iterable:
        # Point is outside the current interval -> creating a new one
        if interval is None or x not in interval:
            interval_offset = (x - origin) // interval_size
            interval_left = origin + interval_offset * interval_size
            if interval_left == x and intervals_right_closed:
                # Intervals are closed on the right -> shifting
                interval_offset -= 1
                interval_left -= interval_size
            if (
                    coalesce and
                    interval is not None and
                    interval.right == interval_left
            ):
                # Adjacent intervals -> Merging
                interval, interval_offset = brackets.pop()
                interval = interval.replace_right(interval_left + interval_size)
            else:
                interval = intervals.Interval(
                    left=interval_left,
                    right=interval_left + interval_size,
                    left_closed=not intervals_right_closed,
                    right_closed=intervals_right_closed,
                )
            brackets.append(Bracket(interval, interval_offset))
        # The index of the current bracket
        bracket_indices.append(len(brackets) - 1)

    if not already_sorted:
        # Applying the permutation of indices to the bracket indices
        new_bracket_indices = [None] * len(bracket_indices)
        for i in range(len(bracket_indices)):
            new_bracket_indices[sorted_indices[i]] = bracket_indices[i]
        bracket_indices = new_bracket_indices
    return brackets, bracket_indices

        #     if coalesce and (interval_offset is not None) and (new_interval_left <= brackets[-1].interval.right):
        #         interval_right = new_interval_left + interval_size
        #         brackets[-1].interval = brackets[-1].interval.replace_right(interval_right)
        #     elif interval_offset is None or new_interval_offset != interval_offset:
        #         interval_offset = new_interval_offset
        #         interval_left = new_interval_left
        #         interval_right = interval_left + interval_size
        #         brackets.append(
        #             Bracket(intervals.Interval(
        #                 interval_left,
        #                 interval_right,
        #                 not intervals_right_closed,
        #                 intervals_right_closed
        #             ), interval_offset))
        #
        # bracket_indices.append(len(brackets) - 1)
    #
    # if not already_sorted:
    #     new_bracket_indices = [None] * len(bracket_indices)
    #     for i in range(len(bracket_indices)):
    #         new_bracket_indices[sorted_indices[i]] = bracket_indices[i]
    #     bracket_indices = new_bracket_indices
    #
    # return brackets, bracket_indices


class FlatStoredArray(abstract(object)):
    """A simple list-storing representation of arrays

    Parameters
    ----------
    args: tuple[Any]
        The arguments, depending on the concrete class. Typically,
        these will be the dimensions of the array to create, which
        will then be converted to the corresponding flat dimension
    """
    def __init__(self, *args):
        """Constructor"""
        self.__count = self._getcount(*args)
        self._data = [None] * self.__count

    @classmethod
    def full(cls, value, *args):
        """Creates a new array with all values set to the provided value

        Parameters
        ----------
        value: T
            The fill value to use
        *args: Any, typically ints
            The dimensions of the array
        """
        out = cls(*args)
        out.setall(xconst(value))
        return out

    @classmethod
    def zeros(cls, *args):
        """Generates a new flat array full of zeros

        Parameters
        ----------
        *args: Any, typically ints
            The dimensions of the array
        """
        return cls.full(0., *args)

    @classmethod
    def ones(cls, *args):
        """Generates a new flat array full of ones

        Parameters
        ----------
        *args: Any, typically ints
            The dimensions of the array
        """
        return cls.full(1., *args)

    @abstractmethod
    def __array__(self):
        """Conversion to a (full) numpy array"""
        pass

    @abstractmethod
    def _getcount(self, *args):
        """Transforms a set of construction arguments
        into the dimensions of the array to use

        Returns
        -------
        __count: int
            The total (flat) size of the array to create
        """
        raise NotImplementedError('Pure virtual method')

    @abstractmethod
    def _keytoindex(self, key):
        """Converts a n-dimensional key to the corresponding
        flat index

        Parameters
        ----------
        key: Any, typically tuple[int]
            The key identifying the slot in the abstract view of the array

        Returns
        -------
        idx: int
            The flat index corresponding to this slot in the array
        """
        raise NotImplementedError('Pure virtual method')

    @abstractmethod
    def _indextokey(self, index):
        """Performs the conversion from a flat index to the corresponding
        key. If several keys map to the same element, the choice is on
        the concrete class: only guarantee is that:

            self._keytoindex(self._indextokey(i)) == i

        Parameters
        ----------
        index: int
            The flat index in the array

        Returns
        -------
        key: Any, typically tuple[int]
            The key identifying the slot in the abstract view of the array
        """
        raise NotImplementedError('Pure virtual method')

    def __getitem__(self, key):
        """Access to an item given its key

        Parameters
        ----------
        key: KeyType
            The key, typically a tuple of integers

        Returns
        -------
        elt: T
            The value stored in the array
        """
        return self._data[self._keytoindex(key)]
    
    def __setitem__(self, key, value):
        """Sets the item at the provided slot

        Parameters
        ----------
        key: KeyType
            The key representing the abstract index of the array
        value: T
            The value of the element to store in the array
        """
        self._data[self._keytoindex(key)] = value
        
    def __len__(self):
        """The total _size_ of the array

        Returns
        -------
        l: int
            The total number of elements in the array
        """
        return self.__count
    
    def __repr__(self):
        """Interpretable representation of the data"""
        return repr(self._data)
    
    def __str__(self):
        """Human-friendly representation of the data"""
        return str(self._data)
    
    def setall(self, iterable):
        """Sets every element in the array to the successive values of
        the iterable

        Parameters
        ----------
        iterable: iterable[T]
            The values to set, in STORAGE order
        """
        for i, v in enumerate(iterable):
            if i >= self.__count:
                return
            self._data[i] = v

    def __iter__(self):
        """Iterates over all stored values in STORAGE order

        Yields
        ------
        value: T
            The next stored value in the array
        """
        return iter(self._data)

    def keys(self):
        """Iterates over the keys of the array, in STORAGE order,
        i.e. each index is mapped to its key and yielded

        Yields
        ------
        key: KeyType
            The keys corresponding to each of the storage slots
        """
        return (self._indextokey(i) for i in irange(self.__count))

    def items(self):
        """Iterates over the pairs (key, value) of the array, in storage order

        Yields
        ------
        item: tuple[KeyType, T]
            The pairs of elements stored in the array
        """
        return ((self._indextokey(i), v) for i, v in enumerate(self._data))


class DiagonalArray(FlatStoredArray):
    """Represents a square symmetrical 2D matrix

    Parameters
    ----------
    dim: int
        The dimension of the matrix
    """
    def __init__(self, dim):
        """Constructor"""
        super(DiagonalArray, self).__init__(dim)
        self.__dim = dim

    def __array__(self):
        """Conversion to a full NumPy array"""
        import numpy as np
        if len(self._data) == 0:
            return np.zeros((0, 0))
        out = np.empty((self._dim, self._dim), dtype=type(self._data[0]))
        for (i, j), v in self.items():
            out[i, j] = out[j, i] = v
        return out

    @property
    def dim(self):
        """The dimension of the matrix"""
        return self.__dim
    
    @classmethod
    def _getcount(cls, dim):
        """The total size of the storage, given a certain dimension"""
        return (dim*dim + dim) // 2
    
    @classmethod
    def _keytoindex(cls, key):
        """Maps a pair (int, int) to the corresponding flat index

        Parameters
        ----------
        key: (int, int)
            The pair of integers representing indices in the 2D matrix

        Returns
        -------
        flat: int
            The corresponding flat storage index
        """
        i, j = key
        if i < j:
            i, j = j, i
        return (i*i + i) // 2 + j
    
    @classmethod
    def _indextokey(self, index):
        """Maps a given flat index to a pair of indices corresponding to the
        indices in the matrix.
        By convention, this returns them as (larger, smaller)

        Parameters
        ----------
        index: int
            The index in the flat storage

        Returns
        -------
        i: int
            The corresponding row number
        j: int
            The corresponding column number
        """
        i = int(math.sqrt(2*index))
        n = (i*i + i) // 2
        j = index - n
        if j < 0:
            i -= 1
            n = (i*i + i) // 2
            j = index - n
        return i, j
    
    @classmethod
    def mindim(cls, count):
        """Convenience method which returns the minimum dimension required to
        store a given flat collection of values

        Parameters
        ----------
        count: int
            The number of elements in flat storage

        Returns
        -------
        dim: int
            The smallest integer such that (dim * (dim + 1) // 2) >= count
        """
        dim = int(math.sqrt(2*count))
        if cls._getcount(dim) < count:
            dim += 1
        return dim
    
    @classmethod
    def create(cls, obj):
        """Creates a new instance from an existing FlatStorageArray

        Parameters
        ----------
        obj: FlatStoredArray
            The object to convert

        Returns
        -------
        res: DiagonalArray instance
            The DiagonalArray using the same underlying storage, but
            representing a 2D symmetrical matrix
        """
        if isinstance(obj, DiagonalArray):
            res = DiagonalArray(obj.dim)
            res.setall(obj)
        elif isinstance(obj, SubdiagonalArray):
            res = DiagonalArray(obj.dim)
            for k, v in obj.items():
                res[k] = v
        else:
            res = DiagonalArray(cls.mindim(len(obj)))
            res.setall(obj)
        return res
    
    def tonumpyarray(self, fill=None, symmetric=False):
        """Creates a full numpy array from these data

        Parameters
        ----------
        fill: float, optional
            The fill value, if only creating a lower-triangular array.
            Default is np.nan
        symmetric: bool, optional
            If false (the default), this will create a lower triangular
            array, and fill all other values with 'fill'. Otherwise, returns
            an array with values being symmetrical

        Returns
        -------
        res: np.ndarray[float64](dim, dim)
            A square array with all values at and below the diagonal filled
            with the values in this.
            If 'symmetric=True', values above the diagonal are symmetrically
            filled as well
            Otherwise, they are filled with 'fill'
        """
        import numpy as np
        if fill is None:
            fill = np.NAN
        res = np.empty((self.__dim, self.__dim))
        idx = 0
        for i in range(self.__dim):
            for j in range(i+1):
                res[i, j] = self._data[idx]
                if symmetric:
                    res[j, i] = res[i, j]
                idx += 1
            if not symmetric:
                res[i, i+1:self.__dim] = fill
        return res


class SubdiagonalArray(FlatStoredArray):
    """Represents a symmetrical matrix where the diagonal is undefined

    Parameters
    ----------
    dim: int
        The dimension of the resulting matrix

    Notes
    -----
    It IS possible to call for the diagonal coefficients of this, but the
    returned value will be nonsense.
    """
    def __init__(self, dim):
        """Constructor"""
        super(SubdiagonalArray, self).__init__(dim)
        self.__dim = dim

    def __array__(self):
        """Converts the matrix to a numpy array

        Returns
        -------
        anti: np.ndarray[float64](dim, dim)
            The symmetrical matrix with these coefficients.
            By convention, diagonal elements are set to 0.
        """
        import numpy as np
        if len(self._data) == 0:
            return np.zeros((0, 0))
        out = np.empty((self.dim, self.dim), dtype=type(self._data[0]))
        out[np.tril_indices(self.dim)] = self._data
        out[np.triu_indices(self.dim)] = self._data
        out[np.arange(self.dim), np.arange(self.dim)] = 0.
        return out

    @property
    def dim(self):
        """The dimension of the matrix"""
        return self.__dim
    
    @classmethod
    def _getcount(cls, dim):
        """Maps a given dimension to the number of storage slots to create

        Parameters
        ----------
        dim: int
            The dimension of the 2d matrix to represent

        Returns
        -------
        count: int
            The number of non-diagonal points in the matrix (dim x dim)
        """
        return (dim*dim - dim) // 2
        
    @classmethod
    def _keytoindex(cls, key):
        """Maps a key to the corresponding flat index

        Parameters
        ----------
        key: (int, int)
            The 2d index of the matrix

        Returns
        -------
        flat: int
            The flat index corresponding to this key
        """
        i, j = key
        if i < j:
            i, j = j, i
        return (i*i - i) // 2 + j

    @classmethod
    def _indextokey(cls, index):
        """Maps a given flat index to a pair of indices corresponding to the
        indices in the matrix.
        By convention, this returns them as (larger, smaller)

        Parameters
        ----------
        index: int
            The index in the flat storage

        Returns
        -------
        i: int
            The corresponding row number
        j: int
            The corresponding column number
        """
        i = int(math.sqrt(2*index)) + 1
        n = (i*i - i) // 2
        j = index - n
        if j < 0:
            i -= 1
            n = (i*i - i) // 2
            j = index - n
        return i, j
    
    @classmethod
    def mindim(cls, count):
        """Convenience method which returns the minimum dimension required to
        store a given flat collection of values

        Parameters
        ----------
        count: int
            The number of elements in flat storage

        Returns
        -------
        dim: int
            The smallest integer such that (dim * (dim - 1) // 2) >= count
        """
        dim = int(math.sqrt(2*count)) + 1
        if cls._getcount(dim) < count:
            dim += 1
        return dim
    
    @classmethod
    def create(cls, obj):
        """Creates a new instance from an existing FlatStorageArray

        Parameters
        ----------
        obj: FlatStoredArray
            The object to convert

        Returns
        -------
        res: SubdiagonalArray instance
            The SubdiagonalArray using the same underlying storage, but
            representing a 2D symmetrical matrix
        """
        if isinstance(obj, SubdiagonalArray):
            res = SubdiagonalArray(obj.dim)
            res.setall(obj)
        elif isinstance(obj, DiagonalArray):
            res = SubdiagonalArray(obj.dim)
            for k, v in obj.items():
                if k[0] != k[1]:
                    res[k] = v
        else:
            res = SubdiagonalArray(cls.mindim(len(obj)))
            res.setall(obj)
        return res

    def tonumpyarray(self, fill=None, symmetric=False):
        """Creates a full numpy array from these data

        Parameters
        ----------
        fill: float, optional
            The fill value for the diagonal, and above if only creating a
            lower-triangular array. Default is np.nan
        symmetric: bool, optional
            If false (the default), this will create a lower triangular
            array, and fill all other values with 'fill'. Otherwise, returns
            an array with values being symmetrical

        Returns
        -------
        res: np.ndarray[float64](dim, dim)
            A square array with all values below the diagonal filled
            with the values in this.
            If 'symmetric=True', values above the diagonal are symmetrically
            filled as well
            Otherwise, they are filled with 'fill'
            In any case, diagonal is filled with 'fill'
        """
        import numpy as np
        if fill is None:
            fill = np.NAN
        res = np.empty((self.__dim, self.__dim))
        idx = 0
        for i in range(self.__dim):
            for j in range(i):
                res[i, j] = self._data[idx]
                if symmetric:
                    res[j, i] = res[i, j]
                idx += 1
            res[i, i] = fill
            if not symmetric:
                res[i, i+1:self.__dim] = fill
        return res
