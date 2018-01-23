from abc import ABCMeta, abstractmethod
import io
from thalesians.tsa.compat import with_metaclass


class Inter1D(with_metaclass(ABCMeta, object)):
    """Represents a generic 1-dimensional interval"""
    @classmethod
    def new(
            cls, lower=None, upper=None,
            left_closed=True, right_closed=True, closed=None
    ):
        """Generic constructor -> matches arguments to the appropriate subclass

        Parameters
        ----------
        lower: float, optional
            The lower bound of the interval
        upper: float, optional
            The upper bound of the interval, if any
        left_closed: bool, optional
            Whether the left bound is closed; default True (i.e. closed)
        right_closed: bool, optional
            Whether the right bound is closed; default False (i.e. open)
        closed: bool | None, optional
            Whether the interval is closed. If set, this OVERRIDES the values
            of left/right_closed

        Returns
        -------
        cls: Interval instance
             The concrete interval corresponding to this
        """
        if closed is not None:
            left_closed = right_closed = closed
        if lower is None and upper is None:
            return Empty()
        elif upper is None or upper == lower:
            if left_closed and right_closed:
                return Point(lower)
            else:
                return Empty()
        return Interval(
            lower, upper, left_closed=left_closed, right_closed=right_closed
        )

    @property
    @abstractmethod
    def length(self):
        """The length of the interval

        Returns
        -------
        length: float
            The length of the interval, in the same unit this is defined in
        """
        return 0

    def __xor__(self, other):
        """Overrides the '^' XOR operator for intersection

        Parameters
        ----------
        other: Interval instance
            The interval being used

        Returns
        -------
        inter: Interval instance
            The intersection of the two intervals
        """
        return self.intersection(other)

    @abstractmethod
    def intersection(self, other):
        """The intersection of two intervals is always an interval

        Parameters
        ----------
        other: Interval instance
            The other interval this is being intersected with

        Returns
        -------
        inter: Interval instance
            The intersection of the two intervals, as a new interval
        """
        pass

    @abstractmethod
    def __contains__(self, item):
        """Whether a given point is in the interval

        Parameters
        ----------
        item: float
            The item in the interval

        Returns
        -------
        inside: bool
            Whether this value is contained in the provided interval
        """
        return False

    def __eq__(self, other):
        """Compares two intervals"""
        if issubclass(type(other), type(self)):
            return other.equal(self)
        elif issubclass(type(self), type(other)):
            return self.equal(other)
        # Incomparable types !
        return False

    @abstractmethod
    def equal(self, other):
        """Compares an interval with a similar interval

        Parameters
        ----------
        other: Interval instance
            The other interval, of the correct type

        Returns
        -------
        eq: bool
            Whether the two values are equal
        """
        return False


class Empty(Inter1D):
    """The empty interval contains nothing"""
    def __init__(self):
        """Constructor"""
        pass

    @property
    def length(self):
        """By definition, the empty interval has no length"""
        return 0.

    def intersection(self, other):
        """Empty ^ Any -> Empty"""
        return Empty()

    def __contains__(self, item):
        """Empty contains nothing"""
        return False

    def equal(self, other):
        """Empty intervals are all similar"""
        return True

    def __str__(self):
        """Human-readable representation of the empty interval"""
        return '{}'

    def __repr__(self):
        """Interpretable representation of the empty interval"""
        return 'Empty()'


# A simple accessor to an empty interval
empty = Empty()


class Point(Inter1D):
    """Represents a single closed point

    Parameters
    ----------
    value: float
        The value at which the point is defined
    """
    __slots__ = ('value',)

    def __init__(self, value):
        """Constructor"""
        self.value = value

    @property
    def length(self):
        """By definition, a single point has no length"""
        return 0.

    def intersection(self, other):
        """Point ^ I -> Point if Point in I, else Empty"""
        return Point(self.value) if self.value in other else Empty()

    def __contains__(self, item):
        """Contains contains only a single point, its value"""
        return item == self.value

    def equal(self, other):
        """{x} == {y} <=> x == y"""
        return self.value == other.value

    def __str__(self):
        """String representation of the point"""
        return '{{{self.value}}}'.format(self=self)

    def __repr__(self):
        """Interpretable representation of the point"""
        return 'Point({self.value})'.format(self=self)


def point(value):
    """Access to a single point (convenience only)

    Returns
    -------
    pt: Point instance
        The interval corresponding to this point
    """
    return Point(value)


class Interval(Inter1D):
    """Represents a simple interval on the real line

    Parameters
    ----------
    left: float
        The lower bound of the interval
    right: float
        The upper bound of the interval
    left_closed: bool, optional [default=False]
        Whether the interval is closed on the left
    right_closed: bool, optional [default=False]
        Whether the interval is closed on the right
    """
    __slots__ = ('_left', '_right', '_left_closed', '_right_closed', '_str')

    def __init__(self, left, right, left_closed=False, right_closed=False):
        """Constructor"""
        self._left = left
        self._right = right
        self._left_closed = left_closed
        self._right_closed = right_closed
        # String representation of the interval
        self._str = None

    @property
    def length(self):
        """The length of the segment"""
        return self.right - self.left

    def intersection(self, other):
        """[a; b] ^ [c; d] -> [max(a,c); min(b,d)]
        Closedness is given by the 'deciding' boundary, i.e. if a > b, then
        (a; ...} ^ [b; ...} -> (a, ...}
        """
        if not isinstance(other, Interval):
            # Other cases are easier handled in simpler classes
            return other.intersection(self)
        # If self.left == other.left, taking the LEAST closed, since
        # [a; ...} ^ (a; ...} -> (a; ...}
        left = max(
            (self.left, not self.left_closed),
            (other.left, not other.left_closed)
        )
        # Min will already select the least closed in case of a tie
        right = min(
            (self.right, self.right_closed),
            (other.right, other.right_closed)
        )
        return Interval.new(left[0], right[0], not left[1], right[1])

    def __contains__(self, item):
        """Whether a value is in the interval

        Parameters
        ----------
        item: float
            The value to compare

        Returns
        -------
        within: bool
            Whether this value is in this segment
        """
        return (
            not (item < self.left or item > self.right) and
            not (item == self.left and not self.left_closed) and
            not (item == self.right and not self.right_closed)
        )

    @property
    def left(self):
        """Read access to the lower bound of the interval

        Returns
        -------
        left: float
            The left (lower) bound of the interval
        """
        return self._left
        
    @property
    def right(self):
        """Read access to the upper bound of the interval

        Returns
        -------
        right: float
            The right (upper) bound of the interval
        """
        return self._right
    
    @property
    def left_closed(self):
        """Whether the left bound is closed

        Returns
        -------
        closed: bool
            Whether the left (lower) bound of the interval is closed
        """
        return self._left_closed
    
    @property
    def right_closed(self):
        """Whether the right bound is closed

        Returns
        -------
        closed: bool
            Whether the right (upper) bound of the interval is closed
        """
        return self._right_closed
        
    def replace_left(self, new_left, new_left_closed=None):
        """Replaces the lower bound of the interval

        Parameters
        ----------
        new_left: float
            The new value of the lower bound of the interval
        new_left_closed: bool | None
            The new closedness of the boundary, or None to keep the current

        Returns
        -------
        out: Interval instance
            The interval with updated values
        """
        if new_left_closed is None:
            new_left_closed = self._left_closed
        if new_left < self.right:
            return Interval(
                left=new_left,
                right=self.right,
                left_closed=new_left_closed,
                right_closed=self.right_closed
            )
        elif new_left == self.right and self.left_closed and self.right_closed:
            return Point(new_left)
        else:
            return Empty()

    def replace_right(self, new_right, new_right_closed=None):
        """Replaces the lower bound of the interval

        Parameters
        ----------
        new_right: float
            The new value of the upper bound of the interval
        new_right_closed: bool | None
            The new closedness of the boundary, or None to keep the current

        Returns
        -------
        out: Interval instance
            The interval with updated values
        """
        if new_right_closed is None:
            new_right_closed = self._right_closed
        if new_right > self._left:
            return Interval(
                left=self._left,
                right=new_right,
                left_closed=self._left_closed,
                right_closed=new_right_closed
            )
        elif new_right == self.left and self.left_closed and self.right_closed:
            return Point(new_right)
        else:
            return Empty()
    
    def equal(self, other):
        """Checks for equality of two or more intervals

        Parameters
        ----------
        other: Interval instance
            The other interval

        Returns
        -------
        equal: bool
            Whether both interval have exactly the same bounds and closedness
        """
        return (
                self._left == other.left and
                self._right == other.right and
                self._left_closed == other.left_closed and
                self._right_closed == other.right_closed
        )
    
    def __str__(self):
        """Human-friendly representation of the interval

        Returns
        -------
        str: str
            Friendly representation of the interval
        """
        if self._str is None:
            s = io.StringIO()
            s.write('[' if self._left_closed else '(')
            s.write(str(self._left))
            s.write(', ')
            s.write(str(self._right))
            s.write(']' if self._right_closed else ')')
            self._str = s.getvalue()
        return self._str
                
    def __repr__(self):
        """Interpretable representation of itself

        Returns
        -------
        repr: str
            String representation of the interval
        """
        return (
            'Interval(left={self.left}, right={self.right},'
            'left_closed={self.left_closed}, right_closed={self.right_closed})'
            ''.format(self=self)
        )
