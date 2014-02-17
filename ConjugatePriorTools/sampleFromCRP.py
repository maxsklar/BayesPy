import samplingTools
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n', help='Number of customers', type=long)
    parser.add_argument('alpha', type=float, help='Alpha: The concentration parameter')
    args = parser.parse_args()
    number_of_customers = args.n
    alpha = args.alpha
    data = samplingTools.chinese_restaurant_process(number_of_customers, alpha)
    print data


if __name__ == "__main__":
    main()