from pacal import *
import time
if __name__ == "__main__":
    u=TrapezoidalDistr(1,2,4,6)
    #u=NormalDistr()
    u.summary()
    u.plot()
    
    tic=time.time()
    #s=iid_max(u, 3)
    op = log
    #B = exp(log(u)*2)
    B=sign(u-2.5)
    
    B.summary()
    s = B+B 
#    A.summary()
#    s  = NormalDistr() / A / 2.0 + B
#    ## op=max
#    s=op(u,u)
#    s=op(s,u)
#    s=op(s,u)
#    s=op(s,s)
    print s.pdf(1.0) 
    
    #s.summary(show_moments=True)
    print time.time()-tic
    B.plot(color="b")
    s.plot()
    s.summary()
    show()