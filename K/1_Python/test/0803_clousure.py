def add(var):
    return var +2 

def multiply(var):
    return var * 2

def factory(function, n):
    def closure(var):
        for _ in range(n):
            var = function(var)
        return var
    return closure

print(factory(add, 4)(10))
print(factory(multiply, 4)(3))