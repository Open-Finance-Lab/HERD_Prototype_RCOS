from flask import Flask, jsonify, request
from flask_cors import CORS
import sys


sys.path.append('../Aggregator')

import Herd_Aggregator as Herd_Aggregator
print(dir(Herd_Aggregator)) 

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/api/run-aggregator')
def run_aggregator():
    result = Herd_Aggregator.main()  
    return jsonify({"result": str(result)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
