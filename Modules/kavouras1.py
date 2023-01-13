# MODULES AAAAA

import numpy as yeet

def returnfloat(bruh):
    if bruh == "pi" or bruh == "np.pi":
        x = yeet.pi
    else:
        try:
            x = float(bruh)
        except:
            x = bruh
    return x

def get2inputs(one,two,prompt0,prompt1,prompt2):
    q = input(prompt0+" ["+prompt1+"/"+prompt2+"]")
    checker = 0
    while checker == 0:
        if yeet.any(one == q):
            checker = 1
            return checker
        elif yeet.any(two == q):
            checker = 2
            return checker
        else:
            print("Your input wasn't understood")
            q = input("Please try again. ["+prompt1+"/"+prompt2+"]")

def get3inputs(one,two,three,prompt0,prompt1,prompt2,prompt3):
    q = input(prompt0+" ["+prompt1+"/"+prompt2+"/"+prompt3+"]")
    checker = 0
    while checker == 0:
        if yeet.any(one == q):
            checker = 1
            return checker
        elif yeet.any(two == q):
            checker = 2
            return checker
        elif yeet.any(three == q):
            checker = 3
            return checker
        else:
            print("Your input wasn't understood")
            q = input("Please try again. ["+prompt1+"/"+prompt2+"/"+prompt3+"]")
