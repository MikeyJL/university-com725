"""Performs linear regression on the Fuel.csv dataset."""

from os.path import exists

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from pandas import DataFrame

# ---------------------------- Summary of dataset ---------------------------- #

df: DataFrame = pd.read_csv("linear-regression/Fuel.csv")

print("\n======== Head ========\n")
print(df.head())

print("\n======== Features ========\n")
print(", ".join(list(df)))


# --------------------- Plot engine size and CO2 emission -------------------- #

data = df[["ENGINESIZE", "CO2EMISSIONS"]]

print("\n======== Data Summary ========\n")
print(data.describe())


plt.scatter(data[["ENGINESIZE"]].values, data[["CO2EMISSIONS"]].values)
plt.title("Engine Size vs. CO2 Emissions Scatter Plot")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")

ENGINE_CO2_PATH = "linear-regression/figures/engine-co2.png"
if not exists(ENGINE_CO2_PATH):
    plt.savefig(ENGINE_CO2_PATH)
    plt.show()

# ------------------------------- Preprocessing ------------------------------ #

train = data[: int(len(data) * 0.8)]
test = data[int(len(data) * 0.2) :]

engine_size_train = np.array(train[["ENGINESIZE"]])
co2_train = np.array(train[["CO2EMISSIONS"]])

print("\n----- Training Shapes -----\n")
print(f"Engine size shape: {engine_size_train.shape}")
print(f"CO2 shape: {co2_train.shape}")

# -------------------------------- Train model ------------------------------- #

model = linear_model.LinearRegression()
model.fit(engine_size_train, co2_train)

print("\n======== Model stats ========\n")
print(f"Coefficient: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# --------------------- Regression model plot -------------------- #

data = df[["ENGINESIZE", "CO2EMISSIONS"]]

plt.scatter(data[["ENGINESIZE"]].values, data[["CO2EMISSIONS"]].values)
plt.plot(
    data[["ENGINESIZE"]].values,
    model.coef_ * data[["ENGINESIZE"]].values + model.intercept_,
    "-r",
)
plt.title("Engine Size vs. CO2 Emissions with Linear Regression")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")

ENGINE_CO2_REG_PATH = "linear-regression/figures/engine-co2-reg.png"
if not exists(ENGINE_CO2_REG_PATH):
    plt.savefig(ENGINE_CO2_REG_PATH)
    plt.show()

# --------------------------- Getting a predictions -------------------------- #


def get_prediction(x_value):
    """Builds the formula to use for a prediction.

    Args:
        x_value (Union[int, float]): _description_

    Returns:
        list[list[float]]: _description_
    """
    return model.coef_ * x_value + model.intercept_


print("\n======== Prediction ========\n")
print(f"CO2 Emission (from formula): {get_prediction(300)[0][0]:.2f}")
print(f"CO2 Emission (from model): {model.predict([[300]])[0][0]:.2f}")

# -------------------------------- Model Evaluation -------------------------------- #

engine_size_test = np.array(test[["ENGINESIZE"]])
co2_test = np.array(test[["CO2EMISSIONS"]])

print("\n----- Testing Shapes -----\n")
print(f"Engine size shape: {engine_size_test.shape}")
print(f"CO2 shape: {co2_test.shape}")

predictions = model.predict(engine_size_test)

print("\n======== Evaluation ========\n")
print(f"Mean absolute error (MAE): {np.mean(np.absolute(predictions - co2_test)):.2f}")
print(f"Mean squared error (MSE): {np.mean((predictions - co2_test) ** 2):.2f}")
print(f"R2 score: {r2_score(co2_test, predictions):.2f}")
