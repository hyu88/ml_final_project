import csv
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py

#to seperate the sentiment and the sentiment text into two different files
#in order to compute the test precision
df_test = pd.read_csv("../data/processed_pre_iphone.csv", sep=" ")
#df_test = pd.read_csv("../data/processed_full_training_dataset.csv", sep=",",names=['sentiment','sentimenttext'])

raw_text = df_test['sentimenttext'].values
raw_label=df_test['sentiment'].values
print raw_label
# positive=df_test[ df_test['sentiment']== "positive"]
# negative=df_test[ df_test['sentiment']== "negative"]
# neutral=df_test[ df_test['sentiment']== "neutral"]
positive=df_test[ df_test['sentiment']== 0]
negative=df_test[ df_test['sentiment']== 1]
neutral=df_test[ df_test['sentiment']== 2]
# df_test['sentimenttext'].to_csv("../data/test_text.csv", encoding='utf-8', index=False)
# df_test['sentiment'].to_csv("../data/test_label.csv", encoding='utf-8', index=False)

a=len(positive)
b=len(negative)
c=len(neutral)
print len(negative)
print len(positive)
print len(neutral)
print a,b,c


# X = [2,4,6]
# Y =[a,c,b]
#
# plt.bar(X, Y)
#
# plt.xlim(1.5,7)
# plt.ylim(0, 100)
# plt.ylabel("number of tweets")
# plt.bar(2, a, facecolor='#ff9999', edgecolor='black')
# plt.bar(4, c, facecolor='#9999ff', edgecolor='black')
# plt.text(2 + 0.5, a + 4, 'positive' , ha='center', va='bottom')
# plt.text(4 + 0.5, c + 4, 'neutral' , ha='center', va='bottom')
# plt.text(6 + 0.5, b + 4, 'negative' , ha='center', va='bottom')
# plt.title("Distribution of Test Tweets",loc='center')
# plt.show()
# print raw_label.shape

# Data to plot
labels = 'Positive\n'+str(a), 'Negative\n'+str(b), 'Neutral\n'+str(c)
sizes = [a, b, c]
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0, 0, 0 )  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()