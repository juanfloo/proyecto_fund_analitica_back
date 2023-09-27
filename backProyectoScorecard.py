#Importamos librerías

import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

app = Flask(__name__)

#Definición de variables
min_score = 300
max_score = 850

#Cargar los datos
datos = pd.read_csv("credit_risk_dataset.csv", delimiter=",")

# Verificamos datos nulos
datos.isnull().sum()

# Borramos datos nulos
datos = datos.dropna()

# Verificamos datos duplicados
datos.duplicated().sum()

# Borramos datos duplicados
datos = datos.drop_duplicates()

# Realizamos un label encoder para convertir las variables cualitativas en categóricas por niveles
LabelEncoder = LabelEncoder()
datos["person_home_ownership_num"] = LabelEncoder.fit_transform(datos["person_home_ownership"])
datos["loan_intent_num"] = LabelEncoder.fit_transform(datos["loan_intent"])
datos["loan_grade_num"] = LabelEncoder.fit_transform(datos["loan_grade"])
datos["cb_person_default_on_file_num"] = LabelEncoder.fit_transform(datos["cb_person_default_on_file"])

# Ahora eliminamos las variables cualitativas, pues son redundantes.
datos = datos.drop(["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"],axis=1)

# Realizamos un train test split para dividir los datos en datos de entrenamiento y datos de testeo
X = datos.drop("loan_status", axis=1)  # Variables predictoras
y = datos["loan_status"]  # Variable objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

model = LogisticRegression()
model.fit(X_train, y_train)


# Denifimos el modelo como una regresión logística (pues queremos clasificar)

def scale_credit_score(probability, min_prob, max_prob):
    return min_prob + (max_prob-min_prob)*probability

def procesar_datos(data):
    pandas_data = pd.DataFrame(data)
    pandas_data = pandas_data.transpose()
    pandas_data.columns = ["person_age","person_income","person_emp_length","loan_amnt", "loan_int_rate",
                           "loan_percent_income", "cb_person_cred_hist_length","person_home_ownership_num",
                           "loan_intent_num","loan_grade_num","cb_person_default_on_file_num"]

    y_pred = model.predict(pandas_data)[0]

    pandas_data['credit_score'] = model.predict_proba(pandas_data)[:, 1]

    pandas_data['credit_score_scaled'] = scale_credit_score(pandas_data['credit_score'], min_score, max_score)
    pandas_data['credit_score_scaled'] = pandas_data['credit_score_scaled'].astype(int)

    scale_credit_score_int = {
        "credit_score_scaled": int(pandas_data.loc[0,"credit_score_scaled"]),
        "y_pred": int(y_pred)
    }


    return scale_credit_score_int


@app.route('/procesar', methods=['POST'])
def princ_function():
    try:
        data = request.json  # Primero, obtén el JSON de la solicitud

        # Extraer los valores y agregarlos a una lista
        #array_data_request = [data["person_age"],data["person_income"],data["person_emp_length"],data["loan_amnt"],data["loan_int_rate"],data["loan_percent_income"],data["cb_person_cred_hist_length"],data["person_home_ownership_num"],data["loan_intent_num"],data["loan_grade_num"],data["cb_person_default_on_file_num"]]
        
        
        return procesar_datos([23, 15000, 6.0, 4000, 12.14, 0.267, 4, 2, 3, 3, 0])
    except Exception as e:
        return {
            'error': str(e) 
        }

    
if __name__ == '__main__':
    app.run(debug=True)


# print(procesar_datos(user_data))