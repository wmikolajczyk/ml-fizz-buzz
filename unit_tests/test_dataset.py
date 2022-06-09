import pandas as pd
import pytest

from dataset import solve_fizz_buzz, generate_data


@pytest.mark.parametrize(
    "num, expected_result",
    (
        (0, "FizzBuzz"),
        (1, ""),
        (2, ""),
        (3, "Fizz"),
        (4, ""),
        (5, "Buzz"),
        (6, "Fizz"),
        (7, ""),
        (8, ""),
        (9, "Fizz"),
        (10, "Buzz"),
        (11, ""),
        (12, "Fizz"),
        (13, ""),
        (14, ""),
        (15, "FizzBuzz"),
        (16, ""),
        (17, ""),
        (18, "Fizz"),
        (101, ""),
        (101010, "FizzBuzz"),
    ),
)
def test_solve_fizz_buzz(num: int, expected_result: str):
    result = solve_fizz_buzz(num)
    assert result == expected_result


def test_generate_data():
    expected_df = pd.DataFrame(
        {
            "number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "label": ["", "", "Fizz", "", "Buzz", "Fizz", "", "", "Fizz", "Buzz"],
        }
    )
    result_df = generate_data(1, 10)
    pd.testing.assert_frame_equal(result_df, expected_df)
