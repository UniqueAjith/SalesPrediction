from flask import Flask,render_template,request
from flask_cors import cross_origin
import pandas as pd
import joblib

app = Flask(__name__, template_folder="templates")

model_fit = joblib.load(open("./models/RForest_fitted_model.pkl", "rb"))
print("Model Loaded")
imputer = joblib.load(open("./models/imputer.pkl", "rb"))
print("Model Loaded")
scaler = joblib.load(open("./models/scaler.pkl", "rb"))
print("Model Loaded")
encoder = joblib.load(open("./models/encoder.pkl", "rb"))
print("Model Loaded")
input_columns = joblib.load(open("./models/input_columns.pkl", "rb"))
print("Model Loaded")
target_column = joblib.load(open("./models/target_column.pkl", "rb"))
print("Model Loaded")
numeric_columns = joblib.load(open("./models/numeric_columns.pkl", "rb"))
print("Model Loaded")
categorical_columns = joblib.load(open("./models/categorical_columns.pkl", "rb"))
print("Model Loaded")
encoded_columns = joblib.load(open("./models/encoded_columns.pkl", "rb"))
print("Model Loaded")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
		
       
        # ['Item_Weight','Item_Fat_Content','Item_Visibility','Item_MRP','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type','New_Item_Type']

        # Item_Weight
        Item_Weight = (request.form['Weight'])
        # Item_Fat_Content
        Item_Fat_Content = (request.form['Fat_content'])
        # Item_Visibility
        Item_Visibility = (request.form['Visibility'])
        # Item_MRP
        Item_MRP = (request.form['MRP'])
        # Outlet_Identifier
        Outlet_Identifier = (request.form['Outlet_ID'])
        # Outlet_Establishment_Year
        Outlet_Establishment_Year = (request.form['Year'])
        # Outlet_Size
        Outlet_Size = (request.form['Size'])
        # Outlet_Location_Type
        Outlet_Location_Type = (request.form['Location'])
        # Outlet_Type
        Outlet_Type = (request.form['Type'])
        # New_Item_Type
        New_Item_Type = (request.form['Item_type'])
        

        input_lst = [Item_Weight , Item_Fat_Content , Item_Visibility , Item_MRP, Outlet_Identifier ,Outlet_Establishment_Year , Outlet_Size, Outlet_Location_Type, Outlet_Type, New_Item_Type]
								
        new_input = pd.DataFrame({
                "Item_Weight": [Item_Weight],
                "Item_Fat_Content": [Item_Fat_Content],
                "Item_Visibility": [Item_Visibility],
                "Item_MRP": [Item_MRP],
                "Outlet_Identifier": [Outlet_Identifier],
                "Outlet_Establishment_Year": [Outlet_Establishment_Year],
                "Outlet_Size": [Outlet_Size],
                "Outlet_Location_Type": [Outlet_Location_Type],
                "Outlet_Type": [Outlet_Type],
                "New_Item_Type": [New_Item_Type]
        })

        def predict_input(input_df):
            input_df[numeric_columns] = imputer.transform(input_df[numeric_columns])
            input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
            input_df[encoded_columns] = encoder.transform(input_df[categorical_columns])
            X_input = input_df[numeric_columns + encoded_columns]
            pred = model_fit.predict(X_input)[0]
            return pred
        prediction = predict_input(new_input)
        # prediction = round(prediction, 2)
        output = prediction

        if output>0:
                 return {
                    "itemid": request.form["Item_ID"],
                    "outletid": request.form["Outlet_ID"],
                    "prediction": output,
                    "status": "success"
                }
        else:
                return render_template("home.html")
    return render_template("home.html")

if __name__=='__main__':
	app.run()