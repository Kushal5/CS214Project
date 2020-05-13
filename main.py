# CS 412 Project
# Contributers: Kushal Pillay, Gary Wong, Nandita Nishika
# Experiment 1: ARIMA Model on Ebola Cases Sierra Leone

from pandas import read_excel
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def main():
	series = read_excel('EbolaMaster.xlsx', header=0, index_col=0, squeeze=True)
	print(series.head())
	series.plot()
	pyplot.savefig('Sierra-Leone')
	pyplot.show()
	autocorrelation_plot(series)
	pyplot.show()

	X = series.values
	size = int(len(X) * 0.66)
	train, test = X[0:size], X[size:len(X)]
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=(5, 1, 0))
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
	error = mean_squared_error(test, predictions)
	print('Test MSE: %.3f' % error)
	# plot
	pyplot.plot(test)
	pyplot.plot(predictions, color='red')
	pyplot.show()


if __name__ == "__main__":
	main()