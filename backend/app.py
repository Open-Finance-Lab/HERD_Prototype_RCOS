from flask import Flask, jsonify, request
from flask_cors import CORS
import sys

# Add the Aggregator module's directory to the system path
# This allows Python to locate and import modules from '../Aggregator'
sys.path.append('../Aggregator')

# Import the Herd_Aggregator module from the Aggregator directory
import Herd_Aggregator as Herd_Aggregator

# Create a Flask application instance
app = Flask(__name__)
# Enable CORS
CORS(app)  

# End point to call aggregator function
@app.route('/api/run-aggregator')
def run_aggregator():
    result = Herd_Aggregator.main()  
    return jsonify({"result": str(result)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
