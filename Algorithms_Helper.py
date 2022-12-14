from functools import lru_cache
from pyspark.sql import SparkSession, Row
from pyspark import SparkContext

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
    
    
    def binary_search(self, data, target):

        self.data = data
        self.target = target

        if len(data) == 0:
            return -1
        
        left, right = 0, len(data)-1 
        mid = (left + right) // 2

        while data[mid] != target:

            # move left or right index according the location of mid
            if target > data[mid]:
                left = mid
            elif target < data[mid]:
                right = mid

            # Calculate the new mid
            mid = (left + right) // 2

        return data[mid]

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

    def find_min_value(self, data):
        '''
        Create variable called min_index and set it to the first position.
        Iterate through the list, if a value is smaller than min_index
        Then min_index = value
        '''

        self.data = data

        min_index = data[0]

        for i in range(len(data)):

            if data[i] < min_index:
                min_index = data[i]

        return min_index

    def improved_selection_sort(self,data):

        '''
        Find smallest element.
        Exchange it with the first item in the array.

        Find the second smallest element.
        Exchange it with the second item in the array.

        Continue until array is sorted
        '''

        self.data = data

        for i in range(len(data)-1):
            min_index = i

            for j in range(i+1, len(data)):

                if data[j] < data[min_index]:
                    min_index = j

            data[i], data[min_index] = data[min_index], data[i]

        return data


    def selection_sort(self, data):
        '''
        Find smallest element.
        Exchange it with the first item in the array.

        Find the second smallest element.
        Exchange it with the second item in the array.

        Continue until array is sorted
        '''
        self.data = data

        original_data = self.data
        
        def find_min_value(data):

            min_index = data[0]

            for i in range(len(data)):

                if data[i] < min_index:
                    min_index = data[i] 

            return min_index


        def find_index(data, value):

            for i in range(len(data)):
                if data[i] == value:
                    return i
            return None           

        min_value = find_min_value(data)
        min_index = find_index(data, min_value)
        first_item = data[0]  
        data[0] = min_value
        data[min_index] = first_item

        print(data)

        for i in range(1,len(data)):

            print(f'\ni: {i}')
            print(f'Full data: {data}')
            print(f'data to process: {data[i:len(data)]}')

            min_value = find_min_value(data[i:len(data)])
            print(f'min_value: {min_value}')

            min_index = find_index(data, min_value)
            print(f'min_index: {min_index}')

            item = data[i]  

            data[i] = min_value
            print(f'data[{i}]: {data[i]}')
            data[min_index] = item


        print('\nDone')
        print(f'Original Array is: {original_data}')
        print(f'Sorted Array is: {data}')

        return data

    def the_change_problem(self, change, target):

        '''
        Find the min number of coins from a set.
        The sum of coints needs to add up to the target
        '''

        self.change = change
        self.target = target

        coin_list = []
        i = len(change) - 1

        while sum(coin_list) != target and i > 0:

            if change[i] <= target:
                coin_list.append(change[i])
                target = target - change[i]
            else:
                i -= 1
                
        return coin_list


    def find_it(self, seq):

        '''
        Find the number that appears an odd number of times.
        There will always be one one.
        Args:
            - seq(list) - A list of numbers.
        Returns:
            - key(int) - Key of hte number that appeares an odd num of times.
        '''

        self.seq = seq

        dict_counter = dict()

        # Create a dictionary and count appearances
        for i in seq:
            if i in dict_counter:
                dict_counter[i] += 1
            else:
                dict_counter[i] = 1

        # If a value in the dictionary is odd, return its key
        for key, value in dict_counter.items():

            if value % 2 == 1:
                return key

    def tribonacci(self, signature, n):
    
        '''
        Given a list of three items, return a tribonacci array of size n.
        Tribonacci Array = each item is equal to the sum of the previous three items.
        '''
        self.signature = signature
        self.n = n
        
        # Edge Cases
        if n == 0 or len(signature) < 3:
            return []
        
        if n < len(signature):
            return signature[0:n]
        
        tri_sequence = signature

        # Stop when the tribonnacci array reaches the requested size(n)
        while len(tri_sequence) < n:

            # Calculate next_item and append to array
            next_item = signature[len(tri_sequence) - 3] + signature[len(tri_sequence) - 2] + signature[len(tri_sequence) - 1]
            tri_sequence.append(next_item)

        return tri_sequence

    def two_sum(self, data, target):

        '''
        Return the indices of two items.
        The sum of the two items should equal to target.
        
        Args:
            - data(list) 
            - target(int)
        Returns:
            - indices(list)
        '''

        self.data = data
        self.target = target

        indices = []

        for i in range(len(data)):
            for j in range(len(data)):

                if data[i] + data[j] == target and i != j:
                    indices.append((i,j))

        return indices

    def quick_sort(self,data):

        '''
        Sort an Array using Quick Sort.
        Args:
            - data(List) - A sequence of integers.
        Returns:
            - data(List) - An ordered sequence of integers.
        '''

        self.data = data

        less = []
        same = []
        more = []

        if len(data) <= 1:
            return data

        mid = len(data) // 2

        # Create three lists:
        ## 1. less = numbers less than mid
        ## 2. same = numbers same as mid
        ## 3. more = numbers more than mid
        for i in range(len(data)):

            if data[i] < data[mid]:
                less.append(data[i])
            elif data[i] > data[mid]:
                more.append(data[i])
            else:
                same.append(data[i])

        # Recursion - reorder the list(less --> same --> more) recursivly 
        return self.quick_sort(less) + same + self.quick_sort(more)
        
    def fibonnaci_series(self, n):

        '''
        Create a fibonnaci series of size n.
        Args:
            - n(Int) - Size of the series.
        Returns
            - fib(List) - Fibonnaci Series.
        '''
        self.n =n

        if n == 0:
            return 0

        if n == 1:
            return [0,1]

        fib = [0,1]
        
        # Item = sum of the last two items
        for i in range(2,n+1):
            item = fib[i-2] + fib[i-1]
            fib.append(item)

        return fib

    def fibonnaci(self, n):

        '''
        Returning the nth item of a fibonnaci series using Memoization.
        
        Args:
            - n(Int)
        Returns
            - n_item(Int) - The nth item in the series.
        '''
        self.n = n
        if n == 0:
            return 0
        
        if n == 1 :
            return 1

        # Calculating the nth item
        n_item = self.fibonnaci(n-2) + self.fibonnaci(n-1)
        return n_item

    @lru_cache
    def fibonnaci_withCache(self, n):

        '''
        Returning the nth item of a fibonnaci series using Memoization.
        Use functools.iru_cache to cache the previous calculations.
        
        Args:
            - n(Int)
        Returns
            - n_item(Int) - The nth item in the series.
        '''

        self.n = n

        if n == 0:
            return 0
        
        if n == 1:
            return 1

        n_item = self.fibonnaci_withCache(n-2) + self.fibonnaci_withCache(n-1)
        return n_item

    def fibonnaci_withCacheManual(self, n, cache=None):

        '''
        Returning the nth item of a fibonnaci series using Memoization.
        Use a dictionary to cache the previous calculations.
        
        Args:
            - n(Int)
        Returns
            - n_item(Int) - The nth item in the series.
        '''
        self.n = n
        self.cache = cache

        if cache==None:
            cache = {}

        if n in cache:
            return cache[n]
        
        if n == 0 or n == 1:
            return n
        
        n_item = self.fibonnaci_withCacheManual(n-1,cache) + self.fibonnaci_withCacheManual(n-2, cache)
        cache[n] = n_item

        return n_item
   

    def fibonnci_tabulate(self, n):
        '''
        Returning the nth item of a fibonnaci series using Tabulation.
        Use a List to store results from the bottom-up.
        
        Args:
            - n(Int)
        Returns
            - n_item(Int) - The nth item in the series.
        '''
        self.n = n

        base_case = [0,1]

        for i in range(2, n+1):
            base_case.append(self.fibonnci_tabulate(i-2) + self.fibonnci_tabulate(i-1))

        return base_case[n]


class codewars:

    def __init__(self):
        pass

    def maskify(self, number:str) -> str:

        '''
        Replace all the items with '#' except for the last four items.
        '''

        self.number = number

        if len(number) < 4:
            return number

        last_four_digits = number[-4:]
        prefix = '#' * (len(number) - 4)

        new_number = prefix + last_four_digits

        return new_number 

    def maskify_spark(self, input):

        '''
        Replace all the items with '#' except for the last four items.
        '''

        self.input = input

        # Create a spark session
        spark = SparkSession.builder.master('local').getOrCreate()

        # Create an RDD from the input
        rdd = spark.sparkContext.parallelize([input])

        # Flatten the rdd - '12' --> [1,2]
        flat_rdd = rdd.flatMap(lambda x: x)

        # Replace all chars with '#' except for the last four digits
        def replace_characters(char):

            if input.index(char) < (len(input) - 4):
                return '#'
            else:
                return char

        hash_rdd = flat_rdd.map(replace_characters)

        # Concat the solution: [#, #, 1,2] --> '##12'
        solution = ''.join(hash_rdd.collect())

        return solution


    def count_sheeps(self, sheep):

        '''
        Count the number of True values in a list. 
        Any character can be in the list.
        '''
        
        self.sheep = sheep

        number_of_ships = 0

        for i in sheep:

            if '__int__' in dir(i) and i > 0:
                number_of_ships += 1

        return number_of_ships

    def count_sheeps_spark(self, input):

        '''
        Count the number of True values in a list. 
        Any character can be in the list.
        '''

        self.input = input

        # Create a spark session
        spark = SparkSession.builder.master('local').getOrCreate()

        # Create an rdd
        rdd = spark.sparkContext.parallelize(input)

        def convert_items(item):

            '''
            Convert items to either 1 or 0.
            '''
            
            if '__int__' in dir(item) and item > 0:
                return 1
            else:
                return 0

        # Convert items to 1 or 0
        counter_rdd = rdd.map(convert_items)

        # Sum the list
        solution = counter_rdd.reduce(lambda x1, x2: x1 + x2 )

        return solution


    def count_sheeps_sparkSQL(self, input):

        '''
        Count the number of True values in a list. 
        Any character can be in the list.
        '''

        self.input = input
        
        # Create a spark session
        spark = SparkSession.builder.master('local').getOrCreate()

        # Create an rdd
        rdd = spark.sparkContext.parallelize(input)

        def mapper(item):
            return Row(
                    Sheep = item
            )
        # Turn the items in the rdd to Rows
        new_rdd = rdd.map(mapper)

        # Create a dataframe and a temp view
        df = spark.createDataFrame(new_rdd)
        df.createOrReplaceTempView('sheeps')

        # Query to data
        # If a sheep is true, set it to 1, otherwise 0
        # Sum the array of 1's and 0's
        query = '''
        select sum(if(Sheep, 1, 0)) as number_of_ships
        from sheeps;
        '''

        results = spark.sql(query)
        results.show()
        return results
    
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
    