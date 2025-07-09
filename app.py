from flask import Flask, render_template
import os

app = Flask(__name__)

os.chdir("C:/Users/Usuario/OneDrive/surv_flask")

@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)


