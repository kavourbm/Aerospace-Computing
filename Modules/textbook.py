import numpy as yeet

def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]

def swapCols(v,i,j):
    v[:,[i,j]] = v[:,[j,i]]

def gaussPivot(a,b,tol=1.0e-12):
    n = len(b)

    # Set up scale factors
    s = yeet.zeros(n)
    for i in range(n):
        s[i] = max(yeet.abs(a[i,:]))

    for k in range(0,n-1):

        # Row interchange, if needed
        p = yeet.argmax(yeet.abs(a[k:n,k])/s[k:n])+k
        if abs(a[p,k]) < tol:
            print('Matrix is singular. unable to pivot')
            break
        if p != k:
            swapRows(b,k,p)
            swapRows(s,k,p)
            swapRows(a,k,p)

        # Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]
    
    if abs(a[n-1,n-1]) < tol:
        error.err('Matrix is singular')

    #Back substitution
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k]-yeet.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b
