import numpy as yeet
import math

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
            return 0
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
        print('Matrix is singular. Unable to pivot')
        return 0

    #Back substitution
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k]-yeet.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b

def gaussSeidel(a,b,iterEqs,x,tol=1.0e-9):
    omega = 1.0
    k = 10
    p = 1
    for i in range(1,501):
        xOld = x.copy()
        x = iterEqs(a,b,x,omega)
        dx = math.sqrt(yeet.dot(x-xOld,x-xOld))
        if dx < tol:
            return x,i,omega
        if i == k:
            dx1 = dx
        if i == k+p:
            dx2 = dx
            omega = 2.0/(1.0+math.sqrt(1.0-(dx2/dx1)**(1.0/p)))

def iterEqs(a,b,omega):
    n = len(b)
    formulas = yeet.array([])
    for i in range(n):
        formula = np.zeros(n)
        for j in range(n):
            if i != j:
                formula[j] = -omega * a[i,j]/a[i,i]
        formula[i] = 1-omega
        formulas.append(formula)
    return formulas

"""
def gaussSeidel(a,b,x,omega):
    n = len(b)
    for k in range(n):
        xNew = yeet.zeros(n)
        for i in range(n):
            s1 = yeet.dot(a[i,:i],xNew[:i])
            s2 = yeet.dot(a[i,i+1],x[1+1:])
            xNew[i] = (1-omega)*x[i]+omega*(b[i]-s1-s2)/a[i,i]
        x = xNew
    return x
"""

def conjGrad(Av,x,b,tol=1.0e-9):
    n = len(b)
    r = b-Av(x)
    s = r.copy()
    for i in range(n):
        u = Av(s)
        alpha = yeet.dot(s,r)/yeet.dot(s,u)
        x = x+alpha*s
        r = b-Av(x)
        if (math.sqrt(yeet.dot(r,r))) < tol:
            break
        else:
            beta = -yeet.dot(r,u)/yeet.dot(s,u)
            s = r+beta*s
    return x,i

def LUdecomp(a,tol=1.0e-9):
    n = len(a)
    seq = yeet.array(range(n))

    # Set up scale factors
    s = yeet.zeros((n))
    for i in range(n):
        s[i] = max(abs(a[i,:]))

    for k in range(0,n-1):

        # Row interchange, if needed
        p = yeet.argmax(yeet.abs(a[k:n,k])/s[k:n])+k
        if abs(a[p,k]) < tol:
            print('Matrix is singular. Unable to pivot')
            return 0
        if p != k:
            swapRows(s,k,p)
            swapRows(a,k,p)
            swapRows(seq,k,p)

        #Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n]-lam*a[k,k+1:n]
                a[i,k] = lam
    return a,seq

def LUsolve(a,b,seq):
    n = len(a)

    # Rearrange constant vector; store it in [x]
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]

    # Solution
    for k in range(1,n):
        x[k] = x[k] - yeet.dot(a[k,0:k],x[0:k])
    x[n-1] = x[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        x[k] = (x[k]-yeet.dot(a[k,k+1:n],x[k+1:n]))/a[k,k]
    return x

def LUinverse(decomp):
    a, seq = decomp[0], decomp[1]
    n = len(a)
    identity = yeet.identity(n)
    y = yeet.zeros((n, n))
    x = yeet.zeros((n, n))
    inverse = yeet.zeros((n, n))

    # Solve for y using forward substitution
    for i in range(n):
        for j in range(n):
            if i == j:
                y[i][j] = 1
            else:
                y[i][j] = identity[i][j]
                for k in range(i):
                    y[i][j] -= a[i][k] * y[k][j]

    # Solve for x using back substitution
    for i in range(n-1,-1,-1):
        for j in range(n):
            if i == (n - 1):
                x[i][j] = y[i][j] / a[i][i]
            else:
                x[i][j] = y[i][j]
                for k in range(i + 1, n):
                    x[i][j] -= a[i][k] * x[k][j]
                x[i][j] /= a[i][i]

    # Reorder the columns of x to get the inverse
    for i in range(n):
        inverse[:,i] = x[:,seq[i]]
    return inverse

def cramers(a,b):
    n = len(b)

    s = yeet.zeros((n))
    deta = yeet.linalg.det(a)
    if deta == 0:
        print('The system is not invertible')
        return 0
    lam = a.copy()
    for i in range(n):
        lam[:,i] = b[i]
        detai = yeet.linalg.det(lam)
        s[i] = detai/deta
    s = s.reshape(-1,1)
    return s