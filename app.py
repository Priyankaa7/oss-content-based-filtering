import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('oss-content-based-filtering.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/oss',methods=['POST'])
def oss():
    '''
    For rendering results on HTML GUI
    '''
    res = recommendation.results(request.args.get('name'))
    # return jsonify(res)
    
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    return render_template('index.html', oss_text='Repository: $ {}'.format(res))

@app.route('/oss_api',methods=['POST'])
def oss_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    res = recommendation.results(request.args.get('name'))
    # prediction = model.predict([np.array(list(data.values()))])
    # output = prediction[0]
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask,request,jsonify
# from flask_cors import CORS 
# import recommendation

# app = Flask(__name__)
# CORS(app)

# @app.route('/oss-content-based-filteration', methods=['POST'])
# def recommend_oss():
#     res = recommendation.results(request.args.get('name'))
#     return jsonify(res)

# if __name__=='__main__':
#     app.run(port=5000, debug = True)
