import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

data = pd.read_csv("teams.csv")
data = data[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

sns.lmplot(x="athletes", y="medals", data=data, fit_reg=True, ci=None)
sns.lmplot(x="age", y="medals", data=data, fit_reg=True, ci=None)

data = data.dropna()

train = data[data["year"] < 2012].copy()
test = data[data["year"] >= 2012].copy()

reg = LinearRegression()

predictors = ["athletes", "prev_medals"]
reg.fit(train[predictors], train["medals"])

predictions = reg.predict[test[predictors]]

test["predictions"] = predictions
test.loc[test["predictions"] < 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()

