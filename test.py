# importing libraries
import numpy as np
import pandas as pd
# importing backend file
from backend import MethodsForText, MethodsForDataframe

# making object of MethodsForText
sentiment = MethodsForText()
# testing english negative
print(sentiment.predict_sentiment("i AM sad and angry :@", "en"))
# testing roman-urdu postitive
print(sentiment.predict_sentiment("i am happy and Over The MOON!", "en"))

# testing roman-urdu negative
print(sentiment.predict_sentiment("main boht NARAZ houn", "in"))
# testing roman-urdu postitive
print(sentiment.predict_sentiment("main boht khush houn", "in"))

# testing urdu negative
print(sentiment.predict_sentiment("میں تم سے ناراض ہوں", "ur"))
# testing urdu positive
print(sentiment.predict_sentiment("میں تم سے خوش ہوں", "ur"))

# making object of MethodsForDataframe
sentiment2 = MethodsForDataframe()
# dataframe w english text
df_eng = pd.DataFrame(
    np.array([["i am happy"], ["i am SAD"], ["i dont mind"]]), columns=["text"])
print(sentiment2.predict_sentiment(df_eng, "en"))

# dataframe w urdu text
df_ur = pd.DataFrame(
    np.array([["میں تم سے ناراض ہوں"], ["میں تم سے خوش ہوں"]]), columns=["text"])
print(sentiment2.predict_sentiment(df_ur, "ur"))

# dataframe w roman urdu text
df_in = pd.DataFrame(
    np.array([["main boht NARAZ houn"], ["main boht KHUSH houn"]]), columns=["text"])
print(sentiment2.predict_sentiment(df_in, "in"))
