from copy import deepcopy

a = [1,2,3]
id(a)
a = deepcopy(a)
id(a)