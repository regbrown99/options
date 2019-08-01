import math
import datetime

def call_value(strike_price, underlying_price):
    '''Returns the value at expiration of a call option.'''
    x = underlying_price - strike_price
    return max(x,0)

def put_value(strike_price, underlying_price):
    '''Returns the value at expiration of a put option.'''
    x = strike_price - underlying_price
    return max(x,0)

def long_call_pnl(strike_price, underlying_price, trade_price):
    '''Returns the P&L at expiration of a long call'''
    call_val = call_value(strike_price, underlying_price)
    pnl = call_val - trade_price
    return pnl

def long_put_pnl(strike_price, underlying_price, trade_price):
    '''Returns the P&L at expiration of a long put'''
    put_val = put_value(strike_price, underlying_price)
    pnl = put_val - trade_price
    return pnl

def short_call_pnl(strike_price, underlying_price, trade_price):
    '''Returns the P&L at expiration of a short call'''
    call_val = call_value(strike_price, underlying_price)
    pnl = trade_price - call_val
    return pnl

def short_put_pnl(strike_price, underlying_price, trade_price):
    put_val = put_value(strike_price, underlying_price)
    pnl = trade_price - put_val
    return pnl

# Black Scholes Model
class BlackScholesModel:
    '''Black-Scholes assumptions:
        European options (no early exercise)
        No dividend'''
        
    def __init__(self, U: float, E: float, t: float, v: float, r: float):
        self.U = U # underlying price (float)
        self.E = E # exercise price (float)
        self.t = t # time to expiration, in years (float)
        self.v = v # annualized volatility (float)
        self.r = r # risk-free interest rate (float)
        
    def scipy_cdf(x):
        from scipy.stats import norm
        return norm.cdf(x)
     
    def normal_cdf(x):
        '''cumulative distribution function for standard normal'''
        import math
        q = math.erf(x/math.sqrt(2.0))
        return (1.0 + q)/2.0
    
    def Nprime(x):
        exponent = -1.0 * math.pow(x,2) / 2.0
        return (1.0/math.sqrt(2.0*math.pi)) * math.exp(exponent)

    def N(self, x):
        return self.normal_cdf(x)

    def time_to_expiration(exp_date):
        '''Calculates time to expiration in days
        Expiration date must be in format 'YYYY-MM-DD' '''
        today = datetime.date.today()
        format = '%Y-%m-%d'
        e = datetime.datetime.strptime(exp_date, format)
        expiration_date = datetime.date(e.year,e.month,e.day)
        n = expiration_date - today
        nbr_days = int(n.days)
        return nbr_days

    def days_to_yrs(days):
        return days/365

    def h(self):
        term1 = math.log1p(self.U/self.E)
        term2 = self.t*(self.v**2/2.0 + self.r)
        term3 = self.v*math.sqrt(self.t)
        return (term1 + term2) / term3
    
     
    def CallValue(self):
        term1 = self.U * self.N(self.h())
        term2 = self.E * math.exp(-1*self.r*self.t)
        i = self.h() - self.v * math.sqrt(self.t)
        term3 = self.N(i)
        return term1 - (term2 * term3)
    
    def PutValue(self):
        term1 = -1 * self.U * self.N(-1*self.h())
        term2 = self.E * math.exp(-1*self.r*self.t)
        i = self.v * math.sqrt(self.t) - self.h()
        term3 = self.N(i)
        return term1 + (term2 * term3)
    
    # Option Greeks
    
    def call_delta(self):
        return self.N(self.h())
    
    def put_delta(self):
        return -1*self.N(-1*self.h())
    
    def call_gamma(self):
        term1 = self.Nprime(self.h())
        term2 = self.U * self.v * math.sqrt(self.t)
        return term1 / term2
    
    def put_gamma(self):
        return self.call_gamma()
    
    def call_theta(self):
        term1 = self.U * self.v * self.Nprime(self.h())
        term2 = 2.0 * math.sqrt(self.t)
        term3 = self.r * self.E * math.exp(-1*self.r*self.t)
        i = self.h() - self.v * math.sqrt(self.t)
        term4 = self.N(i)
        return ((term1 / term2) + (term3 * term4))
    
    def put_theta(self):
        term1 = self.U * self.v * self.Nprime(self.h())
        term2 = 2.0 * math.sqrt(self.t)
        term3 = self.r * self.E * math.exp(-1*self.r*self.t)
        i = self.v * math.sqrt(self.t) - self.h()
        term4 = self.N(i)
        return ((term1/term2) - (term3 * term4))
    
    def call_vega(self):
        return self.U * math.sqrt(self.t) * self.Nprime(self.h())
    
    def put_vega(self):
        return self.call_vega()
    
    def call_rho(self):
        term1 = self.t * self.E * math.exp(-1*self.r*self.t)
        i = self.h() - self.v * math.sqrt(self.t)
        term2 = self.N(i)
        return term1 * term2
    
    def put_rho(self):
        term1 = -1 * self.t * self.E * math.exp(-1*self.r*self.t)
        i = self.v * math.sqrt(self.t) - self.h()
        term2 = self.N(i)
        return term1 * term2
    
