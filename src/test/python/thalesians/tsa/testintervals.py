from unittest import TestCase
from thalesians.tsa.intervals import *


class TestEmpty(TestCase):
    def test_length(self):
        e = Empty()
        self.assertEqual(e.length, 0)

    def test_inter(self):
        e = Empty()
        p = point(2.3)
        self.assertEqual(e.intersection(p), Empty())

    def test_contains(self):
        self.assertNotIn(2.3, Empty())

    def test_empty(self):
        self.assertEqual(Empty(), empty)


class TestPoint(TestCase):
    def test_length(self):
        # Point is of measure 0
        p = Point(2.3)
        self.assertEqual(p.length, 0)

    def test_inter(self):
        p = Point(2.3)
        q = Point(4.5)
        self.assertEqual(p ^ p, p)
        self.assertEqual(p ^ q, empty)

    def test_contains(self):
        p = Point(2.3)
        self.assertIn(2.3, p)
        self.assertNotIn(2.4, p)


class TestInterval(TestCase):
    def test_length(self):
        s = Interval(0., 1., left_closed=True)
        self.assertEqual(s.length, 1.0)

    def test_inter(self):
        s1 = Interval(0., 1., left_closed=True)
        s2 = Interval(0.5, 1.5, left_closed=False)
        self.assertEqual(s1 ^ s2, Interval(0.5, 1., left_closed=False))
        s3 = Interval(-5., 0.)
        self.assertEqual(s1 ^ s3, empty)
        s4 = Interval(-1., 0., right_closed=True)
        self.assertEqual(s1 ^ s4, Point(0.))

    def test_contains(self):
        s = Interval(0., 1., left_closed=True)
        self.assertIn(0.5, s)
        self.assertIn(0., s)
        self.assertNotIn(1.5, s)
        self.assertNotIn(1.0, s)



