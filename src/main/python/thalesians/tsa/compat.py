from abc import ABCMeta, abstractmethod
from sys import version_info

try:
    from six import with_metaclass
except ImportError:
    # Six is not available: redefining
    def with_metaclass(meta, *bases):
        """Poor man's with_metaclass definition

        Parameters
        ----------
        meta: metaclass
            The metaclass to use
        bases: tuple[types]
            The bases of the class

        Returns
        -------
        _: type instance
            The class with specified metaclass, regardless
            of the version
        """
        return meta('_', bases, {})


def abstract(*bases):
    """Abstract class definition

    Parameters
    ----------
    bases: tuple[type]
        The base(s) for the abstract class defined

    Returns
    -------
    abase: type
        The abstract base for the type to use

    Examples
    --------
    >>> class A(abstract(object)):
    ...     @abstractmethod
    ...     def foo(self):
    ...         pass

    """
    return with_metaclass(ABCMeta, *bases)


if version_info.major == 3:
    irange = range
else:
    irange = xrange
