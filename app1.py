import flask
from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import subprocess

app = flask.Flask(__name__)
app.config["DEBUG"] = True

path_base = "/home/lapinta1/lapinta/"
#path_base_local = "C:/Users/lucas/Desktop/The_Bridge/Git/lapinta/"

####################

#comentario prueba webhook

####################

colaboradores = [
    {"colab_id": 1, "name": "Alba", "city": "Barcelona", "age": 28},
    {"colab_id": 2, "name": "Enrique", "city": "Madrid", "age": 28},
    {"colab_id": 3, "name": "Lucas", "city": "Jaén", "age": 27},
    {"colab_id": 4, "name": "Marton", "city": "Real de Madrid", "age": 31},
    {"colab_id": 5, "name": "ChatGPT", "city": "Internes", "age": 2},
    {"colab_id": 6, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 7, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 8, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 9, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 10, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 11, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 12, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 13, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 14, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 15, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 16, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 17, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 18, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 19, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 20, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 21, "name": "Javier", "city": "Madrid", "age": 31},
    {"colab_id": 22, "name": "Javier", "city": "Madrid", "age": 31}]

####################

@app.route('/', methods=['GET'])
def home():
    
    return "<h1> Manoplas Enrique se me cargó el mensaje  bonito, ahora sus jodéis y os quedáis de este</h2><p> Esta API utiliza un modelo de regresión XGBoost para predecir el ratio de suicidio por 100.000 habitantes. <p> Para realizar una predicción escribe: --> 'https://lapinta1.pythonanywhere.com/api/v1/predict' <p> Para reentrenar el modelo: --> 'https://lapinta1.pythonanywhere.com/api/v1/retrain'</p> <p> Para consultar los colaboradores: --> 'https://lapinta1.pythonanywhere.com/api/v1/colaboradores/all'</p>"
####################

@app.route('/api/v1/colab', methods=['GET'])
def colab():
    return "Esta API ha sido desarrollada por Alba, Enrique, Lucas y Martín"

###################

@app.route('/api/v1/colaboradores/all', methods=['GET'])
def get_colaboradores():

    nombres = [colaborador["name"] for colaborador in colaboradores]

    return jsonify({'colaboradores': nombres})

###################

@app.route('/api/v1/colaboradores', methods=['GET'])
def colab_id():
    if 'colab_id' in request.args:  
        id = int(request.args['colab_id'])
    else:
        return "Error: El colab_id no es válido. Por favor, prueba con uno de los siguientes valores: [1, 2, 3, 4]."
    
    results = []
    for colaborador in colaboradores:
        if colaborador['colab_id'] == id:
            results.append(colaborador)
    return jsonify(results)


###################

@app.route('/api/v1/predict', methods=['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    model = pickle.load(open(path_base + 'best_xgb_model_def.pkl', 'rb')) 
    field_names = ["Year","SuicideCount","CauseSpecificDeathPercentage","Population","GDP","GDPPerCapita","InflationRate","EmploymentPopulationRatio","regionname_num","sex_num","agegroup_num","countryname_num"]
    Year = request.args.get('Year', None)
    SuicideCount = request.args.get('SuicideCount', None)
    CauseSpecificDeathPercentage = request.args.get('CauseSpecificDeathPercentage', None)
    Population = request.args.get('Population', None)
    GDP = request.args.get('GDP', None)
    GDPPerCapita = request.args.get('GDPPerCapita', None)
    InflationRate = request.args.get('InflationRate', None)
    EmploymentPopulationRatio = request.args.get('EmploymentPopulationRatio', None)
    regionname_num = request.args.get('regionname_num', None)
    sex_num = request.args.get('sex_num', None)
    agegroup_num = request.args.get('agegroup_num', None)
    countryname_num = request.args.get('countryname_num', None)

    if None in [Year, SuicideCount, CauseSpecificDeathPercentage, Population, GDP, GDPPerCapita, InflationRate, EmploymentPopulationRatio, regionname_num, sex_num, agegroup_num, countryname_num]:
        return "Args empty, the data are not enough to predict"
    else:
        data_to_predict = [np.int64(Year), np.float64(SuicideCount), np.float64(CauseSpecificDeathPercentage), np.float64(Population), np.float64(GDP),\
                           np.float64(GDPPerCapita), np.float64(InflationRate), np.float64(EmploymentPopulationRatio), np.float64(regionname_num),\
                            np.int64(sex_num), np.int64(agegroup_num), np.int64(countryname_num)]
        data_to_predict = pd.DataFrame([data_to_predict], columns = field_names)

        #prediction = model.predict([[np.int64(Year), np.float64(SuicideCount), np.float64(CauseSpecificDeathPercentage), np.float64(Population), np.float64(GDP), np.float64(GDPPerCapita), np.float64(InflationRate), np.float64(EmploymentPopulationRatio), np.float64(regionname_num), np.int64(sex_num), np.int64(agegroup_num), np.int64(countryname_num)]])
        prediction = model.predict(data_to_predict)
        prediction_float = np.float64(prediction[0])
        return jsonify({'predictions': float(prediction_float)})


###################
    
@app.route('/api/v1/retrain', methods=['GET'])
def retrain(): # Rutarlo al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists(path_base + "data_despliegue/data_suicide_rates_new.csv"):
        data = pd.read_csv(path_base + 'data_despliegue/data_suicide_rates_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['DeathRatePer100K']),
                                                        data['DeathRatePer100K'],
                                                        test_size = 0.20,
                                                        random_state=42)
        model = pickle.load(open(path_base + 'best_xgb_model_def.pkl','rb'))
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['DeathRatePer100K']), data['DeathRatePer100K'])
        pickle.dump(model, open(path_base + 'best_xgb_model_def.pkl', 'wb'))

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"
    
###################

@app.route('/webhook_lapinta', methods=['POST'])
def webhook():
    # Ruta al repositorio en pythonAnywhere
    path_repo = '/home/lapinta1/lapinta' 
    servidor_web = '/var/www/lapinta1_pythonanywhere_com_wsgi.py' 

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if 'repository' in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']
            
            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'El directorio del repositorio no existe'}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull', clone_url], check=True)
                subprocess.run(['touch', servidor_web], check=True) # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f'Se realizó un git pull en el repositorio {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error al realizar git pull en el repositorio {repo_name}'}), 500
        else:
            return jsonify({'message': 'No se encontró información sobre el repositorio en la carga útil (payload)'}), 400
    else:
        return jsonify({'message': 'La solicitud no contiene datos JSON'}), 400

if __name__ == '__main__':
    app.run()
