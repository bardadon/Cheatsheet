from typing import Optional

# Creating a function with type hints.
def greet(name: str) -> str:
    return name

# Creating a function without type hints.
def greet_without_typeHints(name):
    return name

# Creating a function with default type hints.
def greet_withDefault(first_name: str, last_name: Optional[str] = None) -> str:
    return first_name + last_name


greet("Alice")  # Will not raise an error
greet(123)  # Will raise an error. greet expects a string

greet_without_typeHints('Bar') # Will raise an error. No type hints

greet_withDefault('Bar', 'Dadon') # Will raise an error. No type hints
greet_withDefault('Bar') # Will not raise an error. None is accepted
greet_withDefault('Bar', 123) # Will raise an error. Int is not accepted