import math

def poisson_probability(k, lam):
	val = (lam ** k) * math.exp(-lam) / math.factorial(k)
	return round(val,5)