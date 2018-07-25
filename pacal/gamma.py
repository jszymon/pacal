import math

# lgamma function (based on GSL)

def polevl( x, coef, N ):
    i = 0
    ans = coef[i]
    i += 1
    while i <= N:
        ans = ans * x + coef[i]
        i += 1
    return ans


LOGPI = 1.14472988584940017414
LS2PI  =  0.91893853320467274178
MAXLGM = 2.556348e305

A = [8.11614167470508450300E-4,
     -5.95061904284301438324E-4,
     7.93650340457716943945E-4,
     -2.77777777730099687205E-3,
     8.33333333333331927722E-2]
B = [-1.37825152569120859100E3,
     -3.88016315134637840924E4,
     -3.31612992738871184744E5,
     -1.16237097492762307383E6,
     -1.72173700820839662146E6,
     -8.53555664245765465627E5]
C = [1.00000000000000000000E0,
    -3.51815701436523470549E2,
    -1.70642106651881159223E4,
    -2.20528590553854454839E5,
    -1.13933444367982507207E6,
    -2.53252307177582951285E6,
    -2.01889141433532773231E6]




def lgamma(x):
    sgngam = 1
    if math.isnan(x):
        return x
    if math.isinf(x) and x > 0:
        return x
    if math.isinf(x) and x < 0:
        return -x

    if x < -34.0:
        q = -x
        w = lgamma(q)
        p = math.floor(q)
        if p == q:
            return float('inf')
        i = p
        if (i & 1) == 0:
            sgngam = -1
        else:
            sgngam = 1
        z = q - p
        if z > 0.5:
            p += 1.0
            z = p - q
        z = q * math.sin(math.pi * z)
        if z == 0.0:
            return float('inf')
        z = LOGPI - math.log(z) - w;
        return z
    if x < 13.0:
        z = 1.0
        p = 0.0
        u = x
        while u >= 3.0:
            p -= 1.0
            u = x + p
            z *= u
        while u < 2.0:
            if u == 0.0:
                return float('inf')
            z /= u
            p += 1.0
            u = x + p
        if z < 0.0:
            sgngam = -1
            z = -z
        else:
            sgngam = 1
        if u == 2.0:
            return math.log(z)
        p -= 2.0
        x = x + p
        p = x * polevl( x, B, 5 ) / polevl( x, C, 6)
        return math.log(z) + p
    if x > MAXLGM:
        return sgngam * float('inf')
    q = ( x - 0.5 ) * math.log(x) - x + LS2PI
    if x > 1.0e8:
        return q
    p = 1.0/(x*x)
    if x >= 1000.0:
        q += ((   7.9365079365079365079365e-4 * p \
                - 2.7777777777777777777778e-3) * p \
                + 0.0833333333333333333333) / x
    else:
        q += polevl( p, A, 4 ) / x
    return q
