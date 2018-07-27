def merge(left, right):

    result = []

    while len(left) > 0 or len(right) > 0:

        if len(left) > 0 and len(right) > 0:

            if left[0] <= right[0]:

                result.append(left[0])

                left = left[1:]

            else:

                result.append(right[0])

                right = right[1:]

        elif len(left) > 0:

            result.append(left[0])

            left = left[1:]

        elif len(right) > 0:

            result.append(right[0])

            right = right[1:]

    

    return result



def merge_sort(list):

    if len(list) <= 1:
        return list

    mid = int(len(list) / 2)    
    
    print("======================== mid ==================")
    print(mid)
    
    leftList = list[:mid]
    rightList = list[mid:]

    print("======================== 1 ")
    print("left")
    print(leftList)
    print("right")
    print(rightList)

    
    print("mege_sort on Left")
    leftList = merge_sort(leftList)
    
    
    print("mege_sort on Right")
    rightList = merge_sort(rightList)
    
    print("======================== 2 ")
    print("left")
    print(leftList)
    print("right")
    print(rightList)

    result = merge(leftList, rightList)
    print("======================== 3 ")
    print(result)

    return result



''' Test code for sorting algorithm '''

list = [14, 7, 3, 12 , 9, 11, 6 , 2]


newList = merge_sort(list)

print( newList )


