# flask.py
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import numpy as np
from numpy import loadtxt
from keras.models import load_model
from pandas import read_csv

app = Flask(__name__)

# load model
model = load_model('normal_data_model.h5')
# summarize model.z
model.summary()
# load dataset
dataframe = read_csv(r'C:\Users\nCalo\Documents\Automifai\Building the SaaS product\Data_sets\2_weeks_SOH_removed.csv', delimiter=",")
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:20]
Y = dataset[:,20]
# evaluate the model


@app.route('/')
def home():
    return render_template('1_html.html')

@app.route('/predict', methods=['POST']) # https://github.com/nitinkaushik01/Deploy_Machine_Learning_Model_on_Flask_App/blob/master/Flask_Sample_App/app.py
def predict():
    w1= float(request.form['week1'])
    w2= float(request.form['week2'])
    w3= float(request.form['week3'])
    w4= float(request.form['week4'])
    w5= float(request.form['week5'])
    w6= float(request.form['week6'])
    w7= float(request.form['week7'])
    w8= float(request.form['week8'])
    w9= float(request.form['week9'])
    w10= float(request.form['week10'])
    w11= float(request.form['week11'])
    w12= float(request.form['week12'])
    w13= float(request.form['week13'])
    w14= float(request.form['week14'])
    w15= float(request.form['week15'])
    w16= float(request.form['week16'])
    w17= float(request.form['week17'])
    w18= float(request.form['week18'])
    w19= float(request.form['week19'])
    w20= float(request.form['week20'])
    w21= float(request.form['week21'])
    w22= float(request.form['week22'])
    w23= float(request.form['week23'])
    w24= float(request.form['week24'])
    w25= float(request.form['week25'])
    w26= float(request.form['week26'])
    w27= float(request.form['week27'])
    w28= float(request.form['week28'])
    w29= float(request.form['week29'])
    w30= float(request.form['week30'])
    w31= float(request.form['week31'])
    w32= float(request.form['week32'])
    w33= float(request.form['week33'])
    w34= float(request.form['week34'])
    w35= float(request.form['week35'])
    w36= float(request.form['week36'])
    w37= float(request.form['week37'])
    w38= float(request.form['week38'])
    w39= float(request.form['week39'])
    w40= float(request.form['week40'])
    w41= float(request.form['week41'])
    w42= float(request.form['week42'])
    w43= float(request.form['week43'])
    w44= float(request.form['week44'])
    w45= float(request.form['week45'])
    w46= float(request.form['week46'])
    w47= float(request.form['week47'])
    w48= float(request.form['week48'])
    w49= float(request.form['week49'])
    w50= float(request.form['week50'])
    w51= float(request.form['week51'])
    w52= float(request.form['week52'])
    w53= float(request.form['week53'])
    w54= float(request.form['week54'])
    w55= float(request.form['week55'])
    w56= float(request.form['week56'])
    w57= float(request.form['week57'])
    w58= float(request.form['week58'])
    w59= float(request.form['week59'])
    w60= float(request.form['week60'])
    w61= float(request.form['week61'])
    w62= float(request.form['week62'])
    w63= float(request.form['week63'])
    w64= float(request.form['week64'])
    w65= float(request.form['week65'])
    w66= float(request.form['week66'])
    w67= float(request.form['week67'])
    w68= float(request.form['week68'])
    w69= float(request.form['week69'])
    w70= float(request.form['week70'])
    w71= float(request.form['week71'])
    w72= float(request.form['week72'])
    w73= float(request.form['week73'])
    w74= float(request.form['week74'])
    w75= float(request.form['week75'])
    w76= float(request.form['week76'])
    w77= float(request.form['week77'])
    w78= float(request.form['week78'])
    w79= float(request.form['week79'])
    w80= float(request.form['week80'])

    w1_4= (w1+w2+w3+w4)
    w5_8= (w5+w6+w7+w8)
    w9_12= (w9+w10+w11+w12)
    w13_16= (w13+w14+w15+w16)
    w17_20= (w17+w18+w19+w20)
    w21_24= (w21+w22+w23+w24)
    w25_28= (w25+w26+w27+w28)
    w29_32= (w29+w30+w31+w32)
    w33_36= (w33+w34+w35+w36)
    w37_40= (w37+w38+w39+w40)
    w41_44= (w41+w42+w43+w44)
    w45_48= (w45+w46+w47+w48)
    w49_52= (w49+w50+w51+w52)
    w53_56= (w53+w54+w55+w56)
    w57_60= (w57+w58+w59+w60)
    w61_64= (w61+w62+w63+w64)
    w65_68= (w65+w66+w67+w68)
    w69_72= (w69+w70+w71+w72)
    w73_76= (w73+w74+w75+w76)
    w77_80= (w77+w78+w79+w80)


    pred_args = [w1_4,w5_8,w9_12,w13_16,w17_20,w21_24,w25_28,w29_32,w33_36,w37_40,w41_44,w45_48,w49_52,w53_56,w57_60,w61_64,w65_68,w69_72,w73_76,w77_80,
]

    pred_args_arr = np.array([w1_4,w5_8,w9_12,w13_16,w17_20,w21_24,w25_28,w29_32,w33_36,w37_40,w41_44,w45_48,w49_52,w53_56,w57_60,w61_64,w65_68,w69_72,w73_76,w77_80])
    prediction = model.predict(pred_args_arr.reshape(1, 20), batch_size=1)

    return render_template('1_html.html', prediction_text='Stock quantity should be {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
