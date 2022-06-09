import pandas as pd


def solve_fizz_buzz(num: int) -> str:
    result = ""
    if num % 3 == 0:
        result += "Fizz"
    if num % 5 == 0:
        result += "Buzz"
    return result


def generate_data(start_num: int, end_num: int) -> pd.DataFrame:
    X, y = [], []
    for i in range(start_num, end_num + 1):
        X.append(i)
        y.append(solve_fizz_buzz(i))
    return X, y
