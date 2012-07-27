from pacal import *
import time
if __name__ == "__main__":
    u=BetaDistr(4,4)
    #u=NormalDistr()
    u.summary()
    tic=time.time()
    #s=iid_max(u, 3)
    B = u
    B.summary()
    A = u + B 
    A.summary()
    s  = NormalDistr() / A / 2.0 + B
    ## op=max
#    s=op(u,u)
#    s=op(s,u)
#    s=op(s,u)
#    s=op(s,s)
    print s.pdf(1.0) 
    
    #s.summary(show_moments=True)
    print time.time()-tic
    s.plot()
    s.summary()
    show()