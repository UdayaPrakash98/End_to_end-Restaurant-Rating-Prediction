import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pickle as cPickle
import logging
import bz2
#from flask_cors import CORS, cross_origin
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger()

app = Flask(__name__)
Dfile=bz2.BZ2File('model','rb')
model=cPickle.load(Dfile)


@app.route('/')
#@cross_origin()
def home():
    app.logger.info("index page is being called")
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
#@cross_origin()
def predict():
    try:
       app.logger.info("values are received from user as list")
       features = [int(x) for x in request.form.values()]
       final = [np.array(features)]
       app.logger.info(final)
       app.logger.info('received values are converted into array' )
       app.logger.info('model will be loaded')
       prediction = model.predict(final)
       app.logger.info('model enables to predict')
       output = round(prediction[0], 1)
       app.logger.info(output)
       app.logger.info('rating is predicted')
       if output >= 4:
            return render_template('index.html', prediction_text='AWESOME ONE --- Your Expected Rating For a Restaurant with given Features is : {} '.format(output))
       elif output > 3 :
               return render_template('index.html', prediction_text='UHMM NOT BAD --- Your Expected Rating For a Restaurant with given Features is : {} '.format(output))
       else:
              return render_template('index.html', prediction_text='OHH NO --- Your Expected Rating For a Restaurant with given Features is : {} '.format(output))
    except:
        app.logger.error('error occured as format accepts only integers')
        return render_template('index.html', prediction_text='GIVEN VALUES HAS ERRORS (as input accepts  only in Numerical  forms)')


if __name__ == "__main__":
    app.run(debug=True)

