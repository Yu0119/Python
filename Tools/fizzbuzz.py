# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('num', type=int, help="Input loop numbers")
args = parser.parse_args()


def fizzbuzz(numbers):
  """
    Args:
      numbers: ループ回数
    Return:
      results: 結果文字列のリスト
  """
  
  results = list()
  for n in range(numbers):
    result = ''
    if not (n%3==0 or n%5==0):
      result = '{}'.format(n)
    else:
      if n%3==0:
        result += 'Fizz'
      if n%5==0:
        result += 'Buzz'
    results.append(result)

  return results


if __name__ == '__main__':
  
  results = fizzbuzz(args.num)
  for result in results:
    print(result)
