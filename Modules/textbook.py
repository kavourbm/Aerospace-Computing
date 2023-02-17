import numpy as yeet
import math
import matplotlib.pyplot as plt

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

def gaussSeidel(iterEqs,x,tol = 1.0e-9):
    omega = 1.0
    k = 10
    p = 1
    for i in range(1,501):
        xOld = x.copy()
        x = iterEqs(x,omega)
        dx = math.sqrt(yeet.dot(x-xOld,x-xOld))
        if dx < tol:
            return x,i,omega

        # Compute relaxation factor after k+p iterations
        if i == k:
            dx1 = dx
        if i == k + p:
            dx2 = dx
            omega = 2.0/(1.0 + math.sqrt(1.0 - (dx2/dx1)**(1.0/p)))
    print("Gauss-Seidel failed to converge")

def conjGrad(A,x,b,tol=1.0e-9):
    n = len(b)
    r = b-yeet.dot(A,x)
    s = r.copy()
    for i in range(n):
        u = yeet.dot(A,s)
        alpha = yeet.dot(s,r)/yeet.dot(s,u)
        x = x+alpha*s
        r = b-yeet.dot(A,x)
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

def LUdecomp3(c,d,e):
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    return c,d,e

def LUsolve3(c,d,e,b):
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    return b

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
    
def evalPoly(a,xData,x):
    n = len(xData) - 1 # Degree of polynomial
    p = a[n]
    for k in range(1,n+1):
        p = a[n-k] + (x -xData[n-k])*p
    return p

def coeffts(xData,yData):
    m = len(xData) # Number of data points
    a = yData.copy()
    for k in range(1,m):
        a[k:m] = (a[k:m] - a[k-1])/(xData[k:m] - xData[k-1])
    return a

def rational(xData,yData,x):
    m = len(xData)
    r = yData.copy()
    rOld = yeet.zeros(m)
    for k in range(m-1):
        for i in range(m-k-1):
            if abs(x - xData[i+k+1]) < 1.0e-9:
                return yData[i+k+1]
            else:
                c1 = r[i+1] - r[i]
                c2 = r[i+1] - rOld[i+1]
                c3 = (x - xData[i])/(x - xData[i+k+1])
                r[i] = r[i+1] + c1/(c3*(1.0 - c1/c2) - 1.0)
                rOld[i+1] = r[i+1]
    return r[0]

def curvatures(xData,yData):
    n = len(xData) - 1
    c = yeet.zeros(n)
    d = yeet.ones(n+1)
    e = yeet.zeros(n)
    k = yeet.zeros(n+1)
    c[0:n-1] = xData[0:n-1] - xData[1:n]
    d[1:n] = 2.0*(xData[0:n-1] - xData[2:n+1])
    e[1:n] = xData[1:n] - xData[2:n+1]
    k[1:n] =6.0*(yData[0:n-1] - yData[1:n])/(xData[0:n-1] - xData[1:n])-6.0*(yData[1:n] - yData[2:n+1])/(xData[1:n] - xData[2:n+1])
    LUdecomp3(c,d,e)
    LUsolve3(c,d,e,k)
    return k

def evalSpline(xData,yData,k,x):
    def findSegment(xData,x):
        iLeft = 0
        iRight = len(xData)-1
        while 1:
            if (iRight-iLeft) <= 1: return iLeft
            i = (iLeft + iRight)//2
            if x < xData[i]: iRight = i
            else: iLeft = i
    i = findSegment(xData,x)
    h = xData[i] - xData[i+1]
    y = ((x-xData[i+1])**3/h-(x-xData[i+1])*h)*k[i]/6.0-((x-xData[i])**3/h-(x-xData[i])*h)*k[i+1]/6.0+(yData[i]*(x-xData[i+1])-yData[i+1]*(x-xData[i]))/h
    return y

def polyFit(xData,yData,m):
    a = yeet.zeros((m+1,m+1))
    b = yeet.zeros(m+1)
    s = yeet.zeros(2*m+1)
    for i in range(len(xData)):
        temp = yData[i]
        for j in range(m+1):
            b[j] = b[j] + temp
            temp = temp*xData[i]
        temp = 1.0
        for j in range(2*m+1):
            s[j] = s[j] + temp
            temp = temp*xData[i]
    for i in range(m+1):
        for j in range(m+1):
            a[i,j] = s[i+j]
    return gaussPivot(a,b)

def returnPoly(x,poly):
    y = yeet.zeros((len(x)))
    for i in range(len(poly)):
        y = y + poly[i]*x**i
    return y

def stdDev(c,xData,yData):
    def evalPoly(c,x):
        m = len(c) - 1
        p = c[m]
        for j in range(m):
            p = p*x + c[m-j-1]
        return p
    n = len(xData) - 1
    m = len(c) - 1
    sigma = 0.0
    for i in range(n+1):
        p = evalPoly(c,xData[i])
        sigma = sigma + (yData[i] - p)**2
    sigma = math.sqrt(sigma/(n - m))
    return sigma

def plotPoly(xData,yData,coeff,xlab='x',ylab='y'):
    m = len(coeff)
    x1 = min(xData)
    x2 = max(xData)
    dx = (x2 - x1)/20.0
    x = yeet.arange(x1,x2 + dx/10.0,dx)
    y = yeet.zeros((len(x)))*1.0
    for i in range(m):
        y = y + coeff[i]*x**i
    plt.plot(xData,yData,'o',x,y,'-')
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.legend(["Data","Solution"])
    plt.grid(True)
    plt.show()