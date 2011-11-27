from pacal import *
class PacalServer:
    _public_methods_ = ['ci', 'summaryStr', 'summary', 'plot', 'execpacal',  'evalpacal']    
    _reg_progid_ = "Python.pacal"
    _reg_clsid_ = "{19353e9b-964a-4b1c-8cd1-baccc1209af6}"
    
    def summaryStr(self):
        if isinstance(self.ans, distr.Distr):
            summ = self.ans.summary_map()
            str = " " + self.ans.getName() +"\n"
            for i in ['mean', 'std', 'var', 'tailexp', 'median', 'medianad', 'iqrange(0.025)',  'range', 'ci(0.05)', 'int_err']:
                if summ.has_key(i): 
                    str += '{0:{align}20}'.format(i, align = '>') + " = " + "{0}".format(repr(summ[i]))+ "\n"     
            return str
        else:
            return None
        
    def summary(self):
        if isinstance(self.ans, distr.Distr):
            return [self.ans.summary_map().keys(), self.ans.summary_map().values()]
        else:
            return None  
        
    def ci(self, level):
        if isinstance(self.ans, distr.Distr):
            return self.ans.ci(level)
        else:
            return None
    
    def plot(self, **args):
        if isinstance(self.ans, distr.Distr):
            self.ans.plot(**args)
            show()
        else:
            pass
    
    def evalpacal(self, cmd):
        self.ans = eval(cmd)
        if isinstance(self.ans, distr.Distr):
            return None
        else: 
            return self.ans    
    def execpacal(self, cmd):
        exec cmd
        return

if __name__=='__main__':
    print "Registering COM server..."
    import win32com.server.register
#    p = PacalServer()
#    print p.summary()
    win32com.server.register.UseCommandLine(PacalServer)