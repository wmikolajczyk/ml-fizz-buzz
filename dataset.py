import pandas as pd


def solve_fizz_buzz(num: int) -> str:
    result = ""
    if num % 3 == 0:
        result += "Fizz"
    if num % 5 == 0:
        result += "Buzz"
    return result


def generate_data(start_num: int, end_num: int) -> pd.DataFrame:
    df = pd.DataFrame({"number": range(start_num, end_num + 1)})
    df["label"] = df["number"].apply(solve_fizz_buzz)
    return df
