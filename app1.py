from flask import Flask,request,jsonify
from flask_cors import CORS 
import recommendation

app = Flask(__name__)
CORS(app)

@app.route('/oss-content-based-filteration', methods=['POST'])
def recommend_oss():
    res = recommendation.results(request.args.get('name'))
    return jsonify(res)

if __name__=='__main__':
    app.run(port=5000, debug = True)