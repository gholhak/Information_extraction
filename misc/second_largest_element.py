'''
Your desired array is here
Example = [2, 4, 9, 10]
'''

import random
import time

ARRAY = [random.randint(1, 100000) for _ in range(100000)]


class SecondLargestElement:
    def __init__(self):
        pass

    @staticmethod
    def check_for_identical_elements(args):
        temp = []
        for e in args:
            if e not in temp:
                temp.append(e)
        if len(temp) == 1:
            out = True
        else:
            out = False
        return out

    '''
    detect the second largest element by sorting the array ascendingly.
    '''

    @staticmethod
    def solution1(args):
        sle = sorted(args)
        return sle[-2]

    '''
    Detect the second largest element by finding the difference between the largest element 
    in the array with remaining elements. The smallest number is the index of the second largest element.
    '''

    @staticmethod
    def solution2(args):
        diff_holder = []
        arr = []
        for e in args:
            if e != max(args):
                diff_holder.append(max(args) - e)
                arr.append(e)
        return arr[diff_holder.index(min(diff_holder))]


def main():
    sle_obj = SecondLargestElement()
    # check_identical = sle_obj.check_for_identical_elements(ARRAY)

    if len(ARRAY) == 0:
        print('There is no element in your array:' + str(None))
    elif len(ARRAY) == 1:
        print('The largest element is:' + str(ARRAY[0]))
    # elif check_identical:
    # print('please provide an array with diverse elements')
    else:
        start_time = time.time()
        s1 = sle_obj.solution1(ARRAY)
        print('The second largest element using solution 1 is:' + str(s1))
        print("-----%s seconds ----" % (time.time() - start_time))

        start_time = time.time()
        s2 = sle_obj.solution2(ARRAY)
        print('The second largest element using solution 2 is:' + str(s2))
        print("-----%s seconds ----" % (time.time() - start_time))


if __name__ == '__main__':
    main()
