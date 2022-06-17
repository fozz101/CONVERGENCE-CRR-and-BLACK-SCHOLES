# -*- coding: utf-8 -*-
"""

@author: fedig
"""



import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(20000)
class OptionPricing:

    def __init__(self,S0,K,t,n,rf,sigma,iterations=None):
        
        self.S0 = S0
        self.K = K
        self.t = t
        self.n=n
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations
    
    def CRR(self,n,Put_Call):
        dt=self.t/n
        u = np.exp(sigma* np.sqrt(dt))
        d = 1/u
        R = np.exp(rf*dt)
        p=(R-d)/(u-d)
        q=1-p
        S=np.zeros((n+1,n+1))
        S[0,0]=S0
        for i in range(1,n+1):
            S[i,0]=S[i-1,0]*u
            for j in range(1,i+1):
                S[i,j]=S[i-1,j-1]*d
                
        
        #Option value at final node
        V=np.zeros((n+1,n+1))
        for j in range(n+1):
            if Put_Call=="C":
                V[n,j]=max(0,S[n,j]-self.K)
            elif Put_Call=="P":
                V[n,j]=max(0,self.K-S[n,j])
    
        #Option price
        for i in range(n-1,-1,-1):
            for j in range(i+1):
                V[i,j]=max(0,1/R*(p*V[i+1,j]+q*V[i+1,j+1]))
        opt_price=V[0,0]
        
        return opt_price
    def BSMtheorique(self,Put_Call):
        d1=(np.log(self.S0/self.K)+(self.rf+self.sigma**2/2)*t)/(self.sigma*np.sqrt(self.t))
        d2=d1 - self.sigma*np.sqrt(self.t)
        if Put_Call=="C":
            opt_price=self.S0*norm.cdf(d1)-self.K*np.exp(-self.rf*self.t)*norm.cdf(d2)
        elif Put_Call=="P":
            opt_price=-self.S0*norm.cdf(-d1)+self.K*np.exp(-self.rf*self.t)*norm.cdf(-d2)
        
        return opt_price

    
    def call_option_simulation(self):
        dt=self.t/self.n
        option_data = np.zeros([self.iterations, 2])
        rand = np.random.normal(0,1, [1, self.iterations])
        stock_price = self.S0*np.exp(dt*(self.rf - 0.5*self.sigma**2) 
                               + self.sigma * np.sqrt(dt) * rand)
        option_data[:,1] = stock_price - self.K
        S = S0 * np.exp(np.cumsum((self.rf - 0.5 * self.sigma ** 2) * dt 
                            + self.sigma * math.sqrt(dt)*
                            np.random.standard_normal((self.n + 1, self.iterations)), axis=0))
        S[0] = S0
        average = np.sum(np.amax(option_data, axis = 1))/float(self.iterations)
        C0 = math.exp(-self.rf * dt) * sum(np.maximum(S[-1] - K, 0)) / self.iterations
        return np.exp(-1.0*self.rf*dt) * average+1,S,C0

    def put_option_simulation(self):
        dt=self.t/self.n
        
        option_data = np.zeros([self.iterations,2])
        rand = np.random.normal(0,1,[1,self.iterations])
        stock_price = self.S0*np.exp(dt*(self.rf - 0.5*self.sigma**2)
                               + self.sigma * np.sqrt(dt) * rand)
        option_data[:,1] = self.K - stock_price
        S = S0 * np.exp(np.cumsum((self.rf - 0.5 * self.sigma ** 2) * dt 
                            + self.sigma * math.sqrt(dt)*
                            np.random.standard_normal((self.n + 1, self.iterations)), axis=0))
        S[0] = S0

        average = np.sum(np.amax(option_data, axis = 1))/float(self.iterations)

        P0 = math.exp(-self.rf * dt) * sum(K-np.maximum(S[-1], 0)) / self.iterations
        
        return np.exp(-1.0*self.rf*dt) * average-1,S
    
    def affichageS(self):
        S = self.call_option_simulation()[1]
        plt.plot(S[:, :10])
        plt.grid(True)
        plt.xlabel('Steps')
        plt.ylabel('Index level')
        plt.show()
    
    def affichageHist(self):
        S = self.call_option_simulation()[1]
        plt.rcParams["figure.figsize"] = (15,8)
        plt.hist(S[-1], bins=50)
        plt.grid(True)
        plt.xlabel('index level')
        plt.ylabel('frequency')
        plt.show()
        
    def ConvergenceCRRtoBSM(self,stepsNumber):
        values=[[],[]]
        for i in range(1,stepsNumber,5):
            print(self.CRR(i,"P"))
            values[0].append(i)
            values[1].append(self.CRR(i,"P"))
    
        plt.plot(values[0],values[1])

        plt.axhline(y=self.BSMtheorique("P"), color='r')
        plt.show()
    
if __name__ == "__main__":
    
    S0=100.	
    K=100.				
    t=2
    n=2			
    rf=0.05			
    sigma=0.2
    iterations=1000000
    model = OptionPricing(S0,K,t,n,rf,sigma,iterations)
    #print("Call option price with Monte Carlo approach: ", model.call_option_simulation())
    #print("Put option price with Monte Carlo approach: ", model.put_option_simulation())
    #print(model.put_option_simulation()[0])
    #model.affichageS()
    #model.affichageHist()
    model.ConvergenceCRRtoBSM(1000)
    #print(model.BSMtheorique("P"))