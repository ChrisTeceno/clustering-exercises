import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")
from matplotlib import style

style.use("ggplot")


def plot_residuals(y, yhat):
    """plot residuals given y and yhat"""
    residuals = y - yhat
    plt.hlines(0, y.min(), y.max(), ls="--")
    plt.scatter(y, residuals, color="blue")
    plt.ylabel("residual ($y - \hat{y}$)")
    plt.xlabel("y value ($y$)")
    plt.title("Actual vs Residual")
    plt.show()


def regression_errors(y, yhat):
    """return metrics"""
    residuals = y - yhat
    return pd.Series(
        {
            "SSE": (residuals ** 2).sum(),
            "ESS": ((yhat - y.mean()) ** 2).sum(),
            "TSS": ((y - yhat.mean()) ** 2).sum(),
            "MSE": mean_squared_error(y, yhat),
            "RMSE": mean_squared_error(y, yhat) ** 0.5,
        }
    )


def baseline_mean_errors(y):
    """return baseline metrics"""
    # make a series of the baseline value
    mean = pd.Series([y.mean()])
    # repeat the value to make a correctly sized series to match y
    mean = mean.repeat(len(y))
    residuals = y - mean
    return pd.Series(
        {
            "SSE": (residuals ** 2).sum(),
            "MSE": mean_squared_error(y, mean),
            "RMSE": mean_squared_error(y, mean) ** 0.5,
        }
    )


def baseline_median_errors(y):
    """return baseline metrics"""
    # make a series of the baseline value
    median = pd.Series([y.median()])
    # repeat the value to make a correctly sized series to match y
    median = median.repeat(len(y))
    residuals = y - median
    return pd.Series(
        {
            "SSE": (residuals ** 2).sum(),
            "MSE": mean_squared_error(y, median),
            "RMSE": mean_squared_error(y, median) ** 0.5,
        }
    )


def better_than_baseline(y, yhat):
    """compare model results to baseline based on mean"""
    # make a series of the baseline value
    mean = pd.Series([y.mean()])
    # repeat the value to make a correctly sized series to match y
    mean = mean.repeat(len(y))
    rmse_baseline = (mean_squared_error(y, mean) ** 0.5,)
    rmse_model = (mean_squared_error(y, yhat) ** 0.5,)
    is_better = rmse_model < rmse_baseline
    # print result
    print(f"based on RMSE, is the model better: {is_better}")
    # return a boolean to be used in a df
    return is_better


def histograms_of_data(df):
    """this function will look at the distribution of the target and a few other variables"""
    # lets change figure size
    plt.rcParams["figure.figsize"] = (20, 8)
    # pick the columns to plot
    cols = [
        "taxvaluedollarcnt",
        "bedroomcnt",
        "bathroomcnt",
        "calculatedfinishedsquarefeet",
        "age",
    ]
    # run throught the columns and plot the distribution
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1
        # Create subplot.
        # plt.subplot(row X col, where?)
        plt.subplot(1, len(cols), plot_number)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        df[col].hist(bins=20)


def graph_variables_vs_target(df):
    """This function will graph the variables vs the target"""
    # lets look at the relationship between the target variable and the other variables
    # lets change figure size
    plt.rcParams["figure.figsize"] = (20, 8)
    # pick the columns to plot
    cols = ["bedroomcnt", "bathroomcnt", "calculatedfinishedsquarefeet", "age"]
    # plt.subplot(row X col, where?)
    fig, axes = plt.subplots(1, 4, sharey=True)
    # run throught the columns and plot the distribution
    for i, col in enumerate(cols):

        # Title with column name.
        axes[i].set_title(col)
        # Display lmplot for column.
        sns.regplot(
            data=df,
            x=col,
            y="taxvaluedollarcnt",
            line_kws={"color": "blue"},
            ax=axes[i],
        )


def q1_graph(df):
    """make the graph for question 1"""
    # make a scatter plot of the relationship between calculatedfinishedsquarefeet and taxvaluedollarcnt
    sns.regplot(
        data=df,
        x="calculatedfinishedsquarefeet",
        y="taxvaluedollarcnt",
        line_kws={"color": "blue"},
    )
    # add title
    plt.title("calculatedfinishedsquarefeet vs taxvaluedollarcnt")
    plt.show()


def q2_graph(df):
    """make the graph for question 2"""
    # lets look at the relationship between the bathroom count and the target variable
    # lets change figure size
    plt.rcParams["figure.figsize"] = (20, 8)
    # lets use a box plot to show the relationship between the target variable and the bathroom count
    sns.boxplot(data=df, x="bathroomcnt", y="taxvaluedollarcnt")
    # add title
    plt.title("Tax Value vs. Bathroom Count")
    plt.show()


def q3_graph(df):
    """make the graph for question 3"""
    # lets look at the relationship between the bedroom count and the target variable
    # lets change figure size
    plt.rcParams["figure.figsize"] = (20, 8)
    # lets use a box plot to show the relationship between the target variable and the bedroom count
    sns.boxplot(data=df, x="bedroomcnt", y="taxvaluedollarcnt")
    # add title
    plt.title("Tax Value vs. Bedroom Count")
    plt.show()


def q4_graph(df):
    """make the graph for question 4"""
    # lets look at the relationship between the age and the target variable
    # lets change figure size
    plt.rcParams["figure.figsize"] = (20, 8)
    # lets use a box plot to show the relationship between the target variable and the age
    sns.regplot(data=df, x="age", y="taxvaluedollarcnt", line_kws={"color": "blue"})
    # add title
    plt.title("Tax Value vs. Age")
    plt.show()


def make_rfe_ranking(X_train, y_train, n=3):
    """make the ranking for the rfe"""
    # make a model for RFE
    model = LinearRegression()
    # use recursive feature elimination to select features
    rfe = RFE(model, n_features_to_select=n)
    # fit the RFE to the training with only the original and not the scaled features
    rfe.fit(X_train, y_train)
    return pd.DataFrame({"rfe_ranking": rfe.ranking_}, index=X_train.columns,)

