# importing libraries
from flask import Flask, render_template, request
# importing backend file
from backend import MethodsForText

app = Flask(__name__)

# making object of class MethodForText
sentiment = MethodsForText()

@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html")

@app.route("/sentimentpredictor")
def sentimentpredictor():
    return render_template("sentimentpredictor.html")

@app.route("/prediction", methods=["POST"])
def prediction():

    # getting text from webpage
    text = request.form.get('textbox')
    if(text == ""):
        return render_template("prediction.html")
    print(text)
    # getting selected language from webpage
    language = request.form.get('language')
    print(language)
    # printing sentiment 0/1 in console
    df = sentiment.predict_sentiment(text, language)

    result_MNB = df['result_MNB'].values
    result_MNB = ' '.join([str(elem) for elem in result_MNB]) 
    result_CNB = df['result_CNB'].values
    result_CNB = ' '.join([str(elem) for elem in result_CNB]) 
    result_LR = df['result_LR'].values
    result_LR = ' '.join([str(elem) for elem in result_LR]) 
    result_SGD = df['result_SGD'].values
    result_SGD = ' '.join([str(elem) for elem in result_SGD]) 
    result_SVC = df['result_SVC'].values
    result_SVC = ' '.join([str(elem) for elem in result_SVC]) 
    prediction = df['prediction'].values
    prediction = ' '.join([str(elem) for elem in prediction])
    print(prediction)
    sum = int(result_MNB) + int(result_CNB) + int(result_LR) + int(result_SGD) + int(result_SVC)
    print(sum)
    if (sum >= 3):
        if(sum==3):
            negative = 60
        if(sum==4):
            negative = 80
        if(sum==5):
            negative = 100
        return render_template("prediction.html", input_text=text, input_lang=language, sentiment=prediction, negative=negative)
    else:
        if(sum==2):
            positive = 60
        if(sum==1):
            positive = 80
        if(sum==0):
            positive = 100
        return render_template("prediction.html", input_text=text, input_lang=language, sentiment=prediction, positive=positive)

@app.route("/sentimentanalysis")
def sentimentanalysis():
    return render_template("sentimentanalysis.html")

if __name__ == "__main__":
    app.run()
    # TEMPLATES_AUTO_RELOAD = True
