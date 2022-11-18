class my_algorithms:
    
    def __init__(self):
        pass
    
    def insertion_sort(self, A):
        
        '''
        Sort an Array using Insertion Sort.
        
        Args:
            - A(list) - A sequence of integers to be sorted.
            
        Returns:
            - A(list) - An ordered sequence.
        '''
        # Raise error if A is not a list
        if isinstance(A, list) == False:
            raise TypeError('The input has to be a list!!!')
        else:
            self.A = A
        
        # Iterate through the array, starting from the second item
        for j in range(1, len(A)):
            
            # i is always one index lower than j
            # So A[j] should always be bigger than A[i], IF the array is sorted.
            key = A[j]
            i = j - 1
            
            # go though the array, and while A[i] is bigger than A[j], move A[i] one spot to the left.
            # Keep doing it until A[j] is bigger than A[i]
            while i > 0 and A[i] > key:
                A[i + 1] = A[i]
                i = i - 1
                
            # Set A[i] to be the index after A[j](i.e skip the part of the array that we already sorted).    
            A[i + 1] = key
            
        return A
    
    
    
    def insertion_sort_desc(self, A):
        
        '''
        Sort an Array using Insertion Sort in Descending Order.
        
        Args:
            - A(list) - A sequence of integers to be sorted.
            
        Returns:
            - A(list) - An ordered sequence.
        '''
        # Raise error if A is not a list
        if isinstance(A, list) == False:
            raise TypeError('The input has to be a list!!!')
        else:
            self.A = A
        
        # Iterate through the array, starting from the third item
        for j in range(2, len(A)):
            
            # i is always one index lower than j
            # So A[j] will always be bigger than A[i], IF the array is sorted.
            key = A[j]
            i = j - 1
            
            # go though the array, and while A[i] is bigger than A[j], move A[i] one spot to the left.
            # Keep doing it until A[j] is bigger than A[i]
            while i > 0 and A[i] < key:
                A[i + 1] = A[i]
                i = i - 1
                
            # Set A[i] to be the index after A[j](i.e skip the part of the array that we already sorted).    
            A[i + 1] = key
            
        return A
    
    def insertion_sort(self, A):
        
        '''
        Sort an Array using Insertion Sort.
        
        Args:
            - A(list) - A sequence of integers to be sorted.
            
        Returns:
            - A(list) - An ordered sequence.
        '''
        # Raise error if A is not a list
        if isinstance(A, list) == False:
            raise TypeError('The input has to be a list!!!')
        else:
            self.A = A
        
        # Iterate through the array, starting from the third item
        for j in range(2, len(A)):
            
            # i is always one index lower than j
            # So A[j] will always be bigger than A[i], IF the array is sorted.
            key = A[j]
            i = j - 1
            
            # go though the array, and while A[i] is bigger than A[j], move A[i] one spot to the left.
            # Keep doing it until A[j] is bigger than A[i]
            while i > 0 and A[i] > key:
                A[i + 1] = A[i]
                i = i - 1
                
            # Set A[i] to be the index after A[j](i.e skip the part of the array that we already sorted).    
            A[i + 1] = key
            
        return A
    
    
    def insertion_sort_desc(self, a):
        
        self.a = a
  
        # traversing the array from 1 to length of array(a)
        for i in range(1, len(self.a)):

            temp = self.a[i]

            # Shift elements of array[0 to i-1], that are
            # greater than temp, to one position ahead
            # of their current position
            j = i-1
            while j >=0 and temp > self.a[j] :
                    self.a[j+1] = self.a[j]
                    j -= 1
            self.a[j+1] = temp
            
        return self.a
    
    def binarySearch(self, nums, target):
        
        self.nums = nums
        self.target = target
        
        """
        :type nums: List[int] -- Must be sorted from left to right
        :type target: int
        :rtype: int
        """
        if len(self.nums) == 0:
            return -1

        left, right = 0, len(self.nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.nums[mid] == self.target:
                return mid
            elif self.nums[mid] < self.target:
                left = mid + 1
            else:
                right = mid - 1

        # End Condition: left > right
        return -1

    def binarySearch_temp2(self, nums, target):

        self.nums = nums
        self.target = target

        """
            :type nums: List[int] -- Must be sorted
            :type target: int
            :rtype: int
        """
        if len(self.nums) == 0:
            return -1

        left, right = 0, len(self.nums)
        while left < right:
            mid = (left + right) // 2
            if self.nums[mid] == self.target:
                return mid
            elif self.nums[mid] < self.target:
                left = mid + 1
            else:
                right = mid

            # Post-processing:
            # End Condition: left == right
        if left != len(self.nums) and nums[left] == self.target:
            return left
        return -1   
        
        
    def linear_search(self, data, target):
        
        '''
        Brute Force Algorithm - Linear Search
        '''

        self.data = data
        self.target = target

        for i in self.data:
            if i == self.target:
                return i
    
  
    
class leetcode:
    
    def __init__(self):
        pass
    
    
    def contains_duplicate(self, nums):
        
        '''
        Return True if there are any duplicates.
            
        Args:
            - nums(list)
            
        Returns:
            - True/False
        '''
        
        self.nums = nums
        
        for i in range(0, len(nums) - 1):
            for j in range(1, len(nums)):

                if i != j:
                    current_item = nums[i]
                    next_item = nums[j]

                    if next_item == current_item:
                        return True
        return False
    
    
    
    
    def best_time_buy(self, prices):
        
        self.prices = prices
        
        if len(prices) < 2:
            return 0
        
        profit_dict = dict()
        profit_dict_temp = dict()
        
        for j in range(0,len(prices)-1) :
            buy_price = prices[j] 
            i = j + 1
            
            
            profit_dict_temp = dict()
            while buy_price < prices[i] and i < len(prices)-1:
                sell_price = prices[i]
                profit = sell_price - buy_price # calculate profit
                profit_dict_temp[i] = profit
                i = i + 1

            if len(profit_dict_temp) > 0:
                profit_dict[j] = max(profit_dict_temp.values())
            else:
                profit_dict[j] = 0
            
        return max(profit_dict.values())

    
    def two_sum_brute_force(self, nums, target):

        self.nums = nums
        self.target = target
        
        print('Brute Force Solution')
        print('\nIterating thorugh all of the array untill finding the result')

        
        if target == 0:
            result = 1
        else:
            result = 0
            
        counter_for = 0
        counter_while = 0
        for j in range(0, len(nums) - 1):
            
            print('\niter #{}'.format(counter_for ))
            
            i = j + 1 # i = 1
            previous_number = nums[j] # 3
            
            print('i = {}, j = {}, previous_number = {}'.format(i, j, previous_number ))

            
            while result != target and i < len(nums):
                print('\nStarting while loop number {}'.format(counter_while))
                current_number = nums[i] # 3
                result = previous_number + current_number # 5
                print('current_number = {}, result = {} + {} = {}'.format(current_number, current_number, previous_number, result))
                
                i = i + 1 # 2
                counter_while += 1
                
                
            if result == target:
                print('\nFound it!!!')
                return [j, i - 1]
            
            counter_for += 1
        return [j, i]
    
    
    
    def two_sum_improved(self, nums, target):

        self.nums = nums
        self.target = target

        print('Improved Brute Force Solution')
        print('Ordering the array first and stop the search once the result is already bigger than target')
        sorted_nums = sorted(nums)

        if target == 0:
            result = 1
        else:
            result = 0
            
        if len(nums) == 2:
            return [0,1]

        counter_for = 0
        counter_while = 0
        for j in range(0, len(sorted_nums) - 1):

            print('\niter #{}'.format(counter_for ))

            i = j + 1 # i = 1
            previous_number = sorted_nums[j] # 3

            print('i = {}, j = {}, previous_number = {}'.format(i, j, previous_number ))


            while result < target and i < len(sorted_nums):
                print('\nStarting while loop number {}'.format(counter_while))
                current_number = sorted_nums[i] # 3
                result = previous_number + current_number # 5
                print('current_number = {}, result = {} + {} = {}'.format(current_number, current_number, previous_number, result))

                i = i + 1 # 2
                counter_while += 1


            if result == target:
                print('\nFound it!!!')
                original_i = nums.index(current_number)
                original_j = nums.index(previous_number)
                return [original_j, original_i]

            counter_for += 1
        original_i = nums.index(current_number)
        original_j = nums.index(previous_number)
        return [original_j, original_i]
    
    
    
    def two_sum_hash_map_with_explanation(self, nums, target):

        self.nums = nums
        self.target = target
        
        # Create a hashmap(a list of all the values in the array and their location)
        hashmap = {}
        counter = 0
        
        print('Building HashMap...')
        for i in range(len(nums)):
            hashmap[nums[i]] = i            
        
        # Calculating the difference between the target and each number in the array
        print('\nHashMap: {}'.format(hashmap))
        print('The target is: {}'.format(target))
        for i in range(len(nums)):
            print('\niter #{}'.format(counter))
              
            complement = target - nums[i]
            print('Current Number is: {}'.format(nums[i]))
            print()
            print('Difference between target and current number is: {} - {} = {}'.format(target, nums[i],complement))
            
            # If the difference is in the hashmap and it is not the current number, than we have found the result. 
            # In that case, return the difference, and the current number
            if complement in hashmap and hashmap[complement] != i:
                print('complement: {} is in HashMap!!!'.format(complement, hashmap[complement], i))
                return [i, hashmap[complement]] 
            
            print('\n{} is not in the array...moving to the next number'.format(complement))
            counter += 1
    
    def two_sum_hash_map(self, nums, target):

        self.nums = nums
        self.target = target
        
        # Create a hashmap(a list of all the values in the array and their location)
        hashmap = {}        
        for i in range(len(nums)):
            hashmap[nums[i]] = i            
        
        # Calculating the difference between the target and each number in the array
        for i in range(len(nums)):     
            complement = target - nums[i]

            # If the difference is in the hashmap and it is not the current number, than we have found the result. 
            # In that case, return the difference, and the current number
            if complement in hashmap and hashmap[complement] != i:
                return [i, hashmap[complement]] 
    