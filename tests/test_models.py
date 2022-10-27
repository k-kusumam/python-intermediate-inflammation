"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
from inflammation.models import daily_mean
from inflammation.models import daily_min
from inflammation.models import daily_max
from inflammation.models import patient_normalise

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""


    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize(
                "test,expected",
                [
                    ([[0,0],[0,0],[0,0]],[0,0]),
                    ([[1,-1],[2,-2],[2,-6]],[1,-6])
                ])
def test_daily_min(test,expected):
    """Test the min function works for an array of zeros"""


    npt.assert_array_equal(daily_min(np.array(test)),np.array(expected))
    '''test_input = np.array([[0,0],
                           [0,0],
                           [0,0],
                          ])
    test_result = np.array([0,0])

    npt.assert_array_equal(daily_min(test_input),test_result)'''
@pytest.mark.parametrize(
                "test,expected",
                [
                    ([[0,0],[0,0],[0,0]],[0,0]),
                    ([[1,-1],[2,-2],[2,-6]],[2,-1])
                ]
                )
def test_daily_max(test,expected):
    """Test the min function works for an array of zeros"""


    npt.assert_array_equal(daily_max(np.array(test)),np.array(expected))

'''@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]]),
    ])
def test_patient_normalise_simple(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
'''
@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
         # other test cases here, with None for expect_raises
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
		(
            'hello',
            None,
            TypeError,
        ),
        (
            3,
            None,
            TypeError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        )

    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""

    if isinstance(test, list):
       test = np.array(test)
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(patient_normalise(test), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(test),np.array(expected), decimal=2)

'''def test_daily_min_negative():
    from inflammation.models import daily_min

    test_input = np.array([[-1,5],
                           [-2,4],
                           [-6,6]])

    test_result = np.array([-6,4])

    npt.assert_array_equal(daily_min(test_input),test_result)

def test_daily_max_negative():
    from inflammation.models import daily_max

    test_input = np.array([[-1,5],
                           [-2,4],
                           [-6,6]])

    test_result = np.array([-1,6])

    npt.assert_array_equal(daily_max(test_input),test_result)
def test_daily_max_type_error():
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([["Hello","There"],["general","Min"]])

'''
