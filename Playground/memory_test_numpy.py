# importing the library
from memory_profiler import profile
import array

# instantiating the decorator
@profile
def my_func():
    x = array.array('d', tuple(x for x in range(0, 1000)))
    y = array.array('d', [y*100 for y in range(0, 1500)])
    del x
    return y
 
if __name__ == '__main__':
    my_func()