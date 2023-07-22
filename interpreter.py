import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
"""
THIS FILE IS FOR REPRESENTING THE LINEGRAPH IN DATA, INTERPRETING ACTUAL & PREDICTED DATA and COMPARING DATA
"""

class LineGraph:
    def __init__(self):
        self.points = []
        self.std_dev = 0
        self.median = 0
        self.mean = 0

    def set_points(self, points):
        self.points = points

    def interpret_data(self):
        df = pd.DataFrame(self.points, columns=['x', 'y'])

        # Step 2: Data Analysis
        # Calculate basic statistics
        self.mean = df['y'].mean()
        self.median = df['y'].median()
        self.std_dev = df['y'].std()

        # Calculate trend using linear regression
        slope, intercept = np.polyfit(df['x'], df['y'], deg=1)


        # Step 4: Summarization
        summary = f"Mean: {self.mean}\nMedian: {self.median}\nStandard Deviation: {self.std_dev}\nTrend: y = {slope}x + {intercept}"
        print(summary)

        # Assuming you have x and y coordinates as separate arrays
        x = np.array([coord[0] for coord in self.points])
        y = np.array([coord[1] for coord in self.points])

        # Fit a linear regression model
        regression_model = LinearRegression()
        regression_model.fit(x.reshape(-1, 1), y)

        # Generate trend summary
        slope = regression_model.coef_[0]
        intercept = regression_model.intercept_

        if slope > 0:
            trend_summary = "The line shows an increasing trend. This means that the y-values generally increase as the x-values increase, indicating positive correlation or growth."
        elif slope < 0:
            trend_summary = "The line shows a decreasing trend. This means that the y-values generally decrease as the x-values increase, indicating negative correlation or decline."
        else:
            trend_summary = "The line is horizontal. This means that the y-values do not significantly change with respect to the x-values, indicating no correlation or stability."

        # Generate detailed summary
        df = pd.DataFrame({'x': x, 'y': y})
        df['diff'] = df['y'].diff()
        df['change_direction'] = np.where(df['diff'] > 0, 'Increasing', np.where(df['diff'] < 0, 'Decreasing', 'No Change'))
        df['change_direction_lag'] = df['change_direction'].shift()

        local_maxima = df[(df['change_direction_lag'] == 'Decreasing') & (df['change_direction'] == 'Increasing')]
        local_minima = df[(df['change_direction_lag'] == 'Increasing') & (df['change_direction'] == 'Decreasing')]

        global_max = df.loc[df['y'].idxmax()]
        global_min = df.loc[df['y'].idxmin()]

        detailed_summary = ""

        if not local_maxima.empty:
            detailed_summary += "There is a local peak at: "
            detailed_summary += ", ".join(f"({point['x']}, {point['y']})" for _, point in local_maxima.iterrows())
            detailed_summary += ". "

        if not local_minima.empty:
            detailed_summary += "There is a local valley at: "
            detailed_summary += ", ".join(f"({point['x']}, {point['y']})" for _, point in local_minima.iterrows())
            detailed_summary += ". "

        if not df.empty:
            detailed_summary += "The global maximum is at: "
            detailed_summary += f"({global_max['x']}, {global_max['y']}). "

            detailed_summary += "The global minimum is at: "
            detailed_summary += f"({global_min['x']}, {global_min['y']}). "

        # Implications of global, local, and start/end points
        implications = ""
        if global_max['y'] == df['y'].iloc[0]:
            implications += "The line starts with the global maximum, suggesting a decreasing trend initially. "
        if global_min['y'] == df['y'].iloc[-1]:
            implications += "The line ends with the global minimum, suggesting an increasing trend towards the end. "
        if global_max['y'] > df['y'].iloc[-1]:
            implications += "The global maximum is higher than the end point, indicating a decreasing trend towards the end. "
        elif global_max['y'] < df['y'].iloc[-1]:
            implications += "The global maximum is lower than the end point, suggesting an increasing trend towards the end. "
        else:
            implications += "The global maximum coincides with the end point, indicating stability or no significant change towards the end. "

        # Data statistics and analytics
        data_statistics = ""
        mean_y = np.mean(y)
        std_y = np.std(y)

        data_statistics += f"The mean of y-values is: {mean_y:.2f}. "
        data_statistics += f"The standard deviation of y-values is: {std_y:.2f}. "

        if std_y == 0:
            data_statistics += "The y-values are constant, showing no variability in the data. "
        else:
            coefficient_of_variation = (std_y / mean_y) * 100
            data_statistics += f"The coefficient of variation is: {coefficient_of_variation:.2f}%, indicating the level of variability in the data. "

            if coefficient_of_variation < 10:
                data_statistics += "The variability in the data is relatively low. This suggests that the y-values are closely grouped around the mean, indicating a more consistent trend. "
            elif coefficient_of_variation >= 10 and coefficient_of_variation < 30:
                data_statistics += "The variability in the data is moderate. This suggests a moderate level of dispersion in the y-values, indicating some fluctuations in the trend. "
            else:
                data_statistics += "The variability in the data is relatively high. This suggests a significant spread in the y-values, indicating substantial fluctuations in the trend. "
            if mean_y > 0:
                data_statistics += "The mean of y-values is positive. This suggests an overall positive trend or growth in the data. "
            elif mean_y < 0:
                data_statistics += "The mean of y-values is negative. This suggests an overall negative trend or decline in the data. "
            else:
                data_statistics += "The mean of y-values is zero. This indicates no significant trend or change in the data. "

            if std_y < mean_y:
                data_statistics += "The standard deviation is smaller than the mean, indicating relatively low variability around the mean. This suggests a more consistent trend in the data. "
            elif std_y > mean_y:
                data_statistics += "The standard deviation is larger than the mean, indicating relatively high variability around the mean. This suggests fluctuations or significant variations in the trend. "
            else:
                data_statistics += "The standard deviation is equal to the mean, indicating uniform variability around the mean. This suggests a consistent level of fluctuations in the trend. "

        # Combine information
        summary = trend_summary + " " + detailed_summary + " " + implications + " " + data_statistics
        print(summary)
        print(self.points.index([global_max['x'], global_max['y']]))


class Interpreter:
    def __init__(self, actual: LineGraph, predicted: LineGraph):
        self.actual = actual
        self.predicted = predicted
        self.mse = None
        self.mae = None
        self.r2 = None
        self.rmse = None
    
    def obtain_point_metrics(self):
        # Convert points to NumPy arrays for easier calculation
        actual_array = self.actual.points
        predicted_array = self.predicted.points
        # Determine the length difference between the arrays
        length_diff = len(actual_array) - len(predicted_array)

        # Pad the predicted_array with values based on the linear interpolation of neighboring points
        if length_diff > 0:
            for i in range(length_diff):
                index = np.argmin([(actual_array[j][1] - predicted_array[j][1]) ** 2 for j in range(len(predicted_array))])
                x = predicted_array[index][0]
                y = (predicted_array[index][1] + predicted_array[index + 1][1]) / 2
                predicted_array.insert(index + 1, [x, y])
        else:
            predicted_array = predicted_array[:len(actual_array)]  # Trim predicted_array to match actual_array length

        # Convert points to NumPy arrays for easier calculation
        actual_array = np.array(actual_array)
        predicted_array = np.array(predicted_array)

        # Calculate metrics
        self.mse  = mean_squared_error(actual_array, predicted_array)
        self.mae  = mean_absolute_error(actual_array, predicted_array)
        self.r2   = r2_score(actual_array, predicted_array)
        self.rmse = np.sqrt(self.mse)
    
    def show_metrics(self):
        print("MSE:", self.mse  )
        print("MAE:", self.mae  )
        print("R2:", self.r2   )
        print("RMSE:", self.rmse )
