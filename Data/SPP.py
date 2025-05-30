import pandas as pd
import statsmodels.api as sm

PATH = "Enter your path here"

# Load the data from Huggingface file
url = "https://huggingface.co/datasets/danilocorsi/LLMs-Sentiment-Augmented-Bitcoin-Dataset/resolve/main/merged/merged_daily.csv"
df = pd.read_csv(url)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter between June 4, 2018 and September 30, 2018
filtered_df = df[(df['timestamp'] >= "2018-06-04") & (df['timestamp'] <= "2018-09-30")]
filtered_df= filtered_df[['timestamp', 'fng_value', 'cbbi_value']]
filtered_df['Aggregate sentiment'] = (filtered_df['fng_value'] + filtered_df['cbbi_value']) / 2
filtered_df.reset_index(drop = True, inplace = True)

#Orthogonalize the 'Aggregate sentiment' feature with respect to the fear and greed index (fng_value)
X = sm.add_constant(filtered_df['Aggregate sentiment'])
y = filtered_df['fng_value']

model = sm.OLS(y, X)
results = model.fit()
residuals = results.resid
filtered_df['fng_feature'] = residuals

save_path = PATH + "sentiment_data.csv"
filtered_df.to_csv(save_path, index = False)