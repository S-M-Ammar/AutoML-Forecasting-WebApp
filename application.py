from flask import Flask,render_template,request,session,url_for,redirect
from process_file import process
from prediction import start


application  = app = Flask(__name__)
application.config["SESSION_PERMANENT"] = False
application.config["SESSION_TYPE"] = "filesystem"
application.config['SECRET_KEY'] = 'THis Very StRong K3Y to Br#@K ! '

global_params = {}

@application.route('/')
def main_page():
    try:
        return render_template("index.html")
    except:
        return render_template("error.html")


@application.route("/signIn_Up")
def show_sigin_up_page():
    try:
        return render_template("signin.html")
    except:
        return render_template("error.html")


@application.route("/forecasting_1",methods=['POST'])
def show_file_screen():
    try:
        return render_template("upload-2.html")
    except:
        return render_template("error.html")


@application.route('/processFile',methods=['POST'])
def process_file():
    try:
        file = request.files['input-file']
        global_params['File'] = file  
        if(file.content_type=="text/csv" or file.content_type=="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
            df,columns = process(file)
            global_params['data'] = df
            global_params['columns'] = columns
            return render_template("usrIpt.html",columns=columns)

        else:
            return "Invalid File"

    except Exception as e:
        print(e)
        return render_template("error.html")


@application.route("/prediction_process",methods=['POST'])
def start_prediction_process():
    try:
        global_params['forecasting_column'] = request.form["forecasting_column"]
        global_params['date_column'] = request.form['date_column']
        global_params['future_units'] = int(request.form['future_units'])
        global_params['time_period'] = request.form['time_period']
        
        result,future_dates = start(global_params['data'],global_params['forecasting_column'],global_params['date_column'],global_params['time_period'],global_params['future_units'])
        if(len(result)==0):
            return render_template("error.html")    
        
        print(type(result))
        print(type(future_dates))
        return render_template("analytics.html",result=result.tolist(),future_dates=future_dates)
    except Exception as e:
        print(e)
        return render_template("error.html")


if __name__ == '__main__':
    application.run(debug=True)