
def removeDuplicates( nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # unique_data = list(dict.fromkeys(nums))
    unique = list(set(nums))

    # unique = unique.sort()
    return len(unique), unique


nums = [1,1,2]
print(removeDuplicates(nums))
