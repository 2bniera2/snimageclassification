import pandas as pd
import matplotlib.pyplot as plt

# read the classification report CSV file
report_df = pd.read_csv('classification_report.csv', index_col=0)

# extract the accuracy scores
acc_scores = report_df.loc['accuracy']

# create a bar chart
plt.bar(acc_scores.index, acc_scores.values)
plt.title('Accuracy Scores')
plt.xlabel('Class Labels')
plt.ylabel('Accuracy')
plt.show()
