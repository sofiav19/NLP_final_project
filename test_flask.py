from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    print("✅ index route hit")
    return "<h1>Test success</h1>"

if __name__ == '__main__':
    print("🔥 Running test Flask")
    app.run(debug=True, host="0.0.0.0", port=5050)

