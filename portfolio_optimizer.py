import pandas as pd
import numpy as np
import math
import datetime
from WebStockReader import WebReader as web


# List of securities to analyze
_SECURITIES = ['fb', 'amzn', 'nflx', 'googl']

# Change service used to get data (yahoo, google, etc.). TIINGO only currently available
_SERVICE = 'tiingo'

'''
# Options for API call. If your selected service requires an API key for authentication,
# this parameter must be provided here. To set an option to default simply comment it out or delete
# it from the dictionary.
# Options
# 'key' - API authentication key (required for Tiingo, Quandl, etc.)
# 'start' - date to begin analysis. DEFAULT: Jan 1, 2010
# 'end' - date to end analysis. DEFAULT: today
# 'log' - use log returns if true, otherwise simple returns DEFAULT: False
# 'rfr' - define the Risk Free Rate (for calculation of Sharpe ratio). DEFAULT: 0
'''
options = {
	'key': '',
	# 'start': datetime.datetime(2009, 3, 5),
	# 'end': datetime.datetime.now(),
	'log': True,

}


class Optimizer(object):
	'''
	Class for optimizing portfolio. Monte Carlo function return the weights of securities
	that maximize Sharpe ratio. Get Sharpe function returns the sharpe ratio for
	a portfolio of given weights of individual securities
	'''

	def __init__(self, securities, service, **kwargs):

		self.securities = securities
		self.service = service

		'''
		Call function that iterates through the securities and queries API
		Function returns a pandas multiIndexed dataframe with the top column
		describing the type of data (ope, close, volume, adjusted close, etc.).
		Calling a data type in this definition allows for quick analysis
		'''
		self.data = web().get(securities, service, **kwargs)['adjClose']

		# Determine if using log returns or standard, then define returns
		if 'log' in kwargs and kwargs['log']:
			self.ret = np.log(self.data).diff()
		else:
			self.ret = self.data.pct_change()

		# Define covariance matrix based on returns above
		self.cov = self.ret.cov()

		# Define the risk free rate for sharpe ratio
		if 'rfr' in kwargs:
			self.rfr = kwargs['rfr']
		else:
			self.rfr = 0

	def moneCarlo(self, sims, n=0):
		# Have a local reference copy of the securities property
		securities = self.securities
		# Numpy array to hold results from each iteration of the simulation
		# results = np.zeros((sims, len(securities) + 3))
		results = []

		# Iterate over the number of monte carlo simulations provided
		# each iteration represents a random possible weight
		for i in range(sims):
			# Numpy array of randomly generated weights for the securities
			weights = np.array(np.random.random(len(securities)))
			# Normalize these weights so that they add to 100%
			weights /= np.sum(weights)
			# Round the weights to whole percentages
			weights = np.around(weights, decimals=2)
			# Calculate portfolio overall anualized return given daily returns and weights
			portfolio_return = np.sum(self.ret.mean() * weights) * 252
			# Calculate portfolio standard deviaiton given weights and returns
			portfolio_std = math.sqrt(np.dot(weights.T, np.dot(self.cov, weights))) * math.sqrt(252)

			iterres = [portfolio_return, portfolio_std, (portfolio_return - self.rfr) / portfolio_std]
			for j in range(len(weights)):
				iterres.append(weights[j])

			results.append(iterres)

			# # Add the resulting returns and deviations altered by the weights to the 'results' array
			# results[i, 0] = portfolio_return
			# results[i, 1] = portfolio_std
			# # Add the Sharpe ratio to the array
			# results[i, 2] = (results[i, 0] - self.rfr) / results[i, 1]
			# # Add the weights generated to the result array
			# for j in range(weights):
			# 	results[i, j + 3] = weights[j]
		results = np.array(results)
		# print(results)
		# Define colums for resultant pandas dataframe
		cols = ['Returns', 'StdDev', 'Sharpe']
		# Add the security names to the columns
		for security in securities:
			cols.append(security)
		# Define the result pandas Dataframe to be returned
		resultFrame = pd.DataFrame(results, columns=cols)

		# return results
		# If user wants the n maximum returns or the single max sharpe ratio
		# Use the same process for minimized volatility
		if n > 0:
			max_sharpe = resultFrame.nlargest(n, 'Sharpe')
			min_vol = resultFrame.nsmallest(n, 'StdDev')
		else:
			max_sharpe = resultFrame.iloc[resultFrame['Sharpe'].idxmax()]
			min_vol = resultFrame.iloc[resultFrame['Sharpe'].idxmin()]

		return max_sharpe, min_vol


if __name__ == "__main__":
	try:
		max_sharpe = Optimizer(_SECURITIES, _SERVICE, **options).moneCarlo(10)
		print(max_sharpe)
	except Exception as e:
		print(e)
