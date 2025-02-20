# General Preprocessing (Before Choosing a Model)

> These steps are universal and should be performed regardless of the ML model:
> 

### Data Cleaning:

- Handle missing values (e.g., imputation or removal).
- Remove duplicates and correct errors or inconsistencies.
- Identify and handle outliers if they could distort the analysis.

### Exploratory Data Analysis (EDA):

- Analyze distributions, correlations, and patterns in the data.
- Visualize data using plots (e.g., histograms, scatter plots, heatmaps).
- Understand relationships between features and target variables.

### Feature Engineering:

- Create new features from existing ones (e.g., extracting date components like “month” or “year”).
- Transform features into more useful representations (e.g., log transformations for skewed data).

### Encoding Categorical Variables:

- Convert categorical features into numerical representations (e.g., one-hot encoding, label encoding)

### Scaling/Normalization:

- Standardize numerical features to have zero mean and unit variance (e.g., StandardScaler).
- Normalize data to a specific range (e.g., MinMaxScaler)