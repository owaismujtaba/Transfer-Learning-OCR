def sortTwo(nums1, m, nums2, n):
    i, j = 0, 0
    nums1 = nums1[:m]
    nums2 = nums2[:n]
    temp_arr, temp_index = [], -1
    i_size, j_size = len(nums1), len(nums2)
    
    while i<i_size and j<j_size:
        if nums1[i] <= nums2[j] and temp_index<0:
            i += 1
        else:
            if len(temp_arr)>0:
                if nums2[j]<temp_arr[0]:
                    temp_arr.append(nums1[i])
                    temp_index += 1
                    nums1[i] = nums2[j]
                    i += 1
                    j += 1
                else:
                    temp_arr.append(nums1[i])
                    temp_index += 1
                    nums1[i] = temp_arr.pop(0)
                    i = i +1
            else:                
                temp_arr.append(nums1[i])
                nums1[i] = nums2[j]
                temp_index += 1
                i += 1
                j += 1
        
        if i == i_size and temp_arr:
            nums1 = nums1 + temp_arr
            temp_arr, temp_index = [], -1
            i_size = len(nums1)
        if j == j_size:
            nums1 = nums1 + temp_arr
    return nums1 + nums2[j:]
a = []
b = [1]   
print(sortTwo(a,0, b,1))