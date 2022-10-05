"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

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
    from inflammation.models import daily_min

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
    from inflammation.models import daily_max

    npt.assert_array_equal(daily_max(np.array(test)),np.array(expected))


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
