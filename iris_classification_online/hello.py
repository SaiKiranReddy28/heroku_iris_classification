from flask import Flask, render_template, request,redirect
import pickle
target_names=['setosa', 'versicolor', 'virginica']
filename="finalized_model.sav"
loaded_model=pickle.load(open(filename,"rb"))


app=Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

    
@app.route('/predict',methods=["GET","POST"])
def predict():

    
    val0=float(request.form.get("sepal_length"))
    val1=float(request.form.get("sepal_width"))
    val2=float(request.form.get("petal_length"))
    val3=float(request.form.get("petal_width"))
    
    output=loaded_model.predict([[val0,val1,val2,val3]])
    output_text=target_names[output[0]]
    return render_template("home.html",prediction_text="This is "+output_text)
    redirect('/')
    
    
if __name__ == "__main__":
    app.run()