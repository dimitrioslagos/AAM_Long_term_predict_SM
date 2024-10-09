from xgboost import XGBClassifier, plot_importance
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, gamma
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import os
from joblib import dump, load



def weibull_calculation(failure_rate):
    FR = np.array(failure_rate)
    # Fit the Weibull CDF to the data
    params, covariance = curve_fit(weibull_cdf, FR[:,0], FR[:,1],p0=[1,12*10])

    # Extract the estimated parameters
    shape_est, scale_est = params
    return scale_est, shape_est

def weibull_cdf(x, k, lambd):
    return 1 - np.exp(-(x / lambd)**k)

def normal_cdf(x, mean, std):
    return norm.cdf(x, mean, std)

def gamma_cdf(x, a,  scale):
    return gamma.cdf(x, a, scale=scale)

def prepare_input_for_EOL_curve(Meter_historic):

    Meter_data = pd.DataFrame(index=Meter_historic["smart_meter_id"].unique(),
                              columns=["life_duration", "status",'brand','model'])
    time_now = datetime(year=2024,month=1,day=1)


    for id,data in Meter_data.iterrows():
        if (Meter_data.loc[id,"status"]=="ok") | (Meter_data.loc[id,"status"]=="fault"):
            continue
        point = Meter_historic.loc[Meter_historic["smart_meter_id"] == id,"delivery_point_id"].values[0]
        meters = Meter_historic.loc[Meter_historic["delivery_point_id"] == point]
        meters.loc[:, "installation_date"] = pd.DatetimeIndex(meters.installation_date)
        if len(meters)>=2:
            meters = meters.sort_values(by='installation_date', ascending=False)
            meters.reset_index(inplace=True)
            for i in range(1, len(meters)):
                Meter_data.loc[meters.loc[i,"smart_meter_id"],"status"] = "Fault"
                Meter_data.loc[meters.loc[i,"smart_meter_id"], "life_duration"] = (meters.loc[i-1,"installation_date"]
                                                                             -meters.loc[i,"installation_date"])/\
                                                                            pd.Timedelta(days=30.44)
                Meter_data.loc[meters.loc[i,"smart_meter_id"], "Replacement_Date"] = meters.loc[i-1,"installation_date"]
                Meter_data.loc[meters.loc[i,"smart_meter_id"], "model"] = meters.loc[i,'model']
                Meter_data.loc[meters.loc[i,"smart_meter_id"], "brand"] = meters.loc[i,'brand']


            Meter_data.loc[meters.loc[0,"smart_meter_id"], "status"] = "ok"
            Meter_data.loc[meters.loc[0,"smart_meter_id"], "life_duration"] = (time_now - meters.loc[0, "installation_date"]) \
                                                                        / pd.Timedelta(days=30.44)
            Meter_data.loc[meters.loc[0,"smart_meter_id"], "model"] = meters.loc[0, 'model']
            Meter_data.loc[meters.loc[0,"smart_meter_id"], "brand"] = meters.loc[0, 'brand']

        else:
            meters.reset_index(inplace=True)
            Meter_data.loc[meters["smart_meter_id"], "status"] = "ok"
            Meter_data.loc[meters["smart_meter_id"], "life_duration"] = (time_now - meters.loc[0, "installation_date"])\
                                                                        /pd.Timedelta(days=30.44)
            Meter_data.loc[meters.loc[0,"smart_meter_id"], "model"] = meters.loc[0, 'model']
            Meter_data.loc[meters.loc[0,"smart_meter_id"], "brand"] = meters.loc[0, 'brand']
    Meter_data = Meter_data[Meter_data["life_duration"] >= 0]
    return Meter_data

def EOL_curve_fit(input_data):
    dict_cdf_params={}
    input_data.loc[:,"model"] = input_data["model"] +'_'+ input_data["brand"]
    for model in input_data.model.unique():
        data2 = input_data[input_data.model==model]
        dict_cdf_params[model] = {'distribution':[], 'N': data2.shape[0],'F': data2[data2.status=='Fault'].shape[0],'params':{}}
        f_r=[]
        data2 = data2[data2.life_duration>=2]
        for i in range(int(data2.life_duration.max())):
            data3 = data2[data2.life_duration <= i]
            f_r.append([i,data3[data3.status=='Fault'].shape[0]/data2.shape[0]])
        if max(f_r)[1]>=0.01:
            if max(f_r)[1]<=0.2:
                f_r = np.array(f_r)
                l,k = weibull_calculation(f_r)
                dict_cdf_params[model]['distribution']='weibull'
                dict_cdf_params[model]['params']['l'] = l
                dict_cdf_params[model]['params']['k'] = k
            else:
                # Use curve_fit to fit the gamma CDF to the data
                f_r = np.array(f_r)
                dict_cdf_params[model]['distribution'] = 'normal'
                params1, _ = curve_fit(normal_cdf, f_r[:, 0], f_r[:, 1], p0=[12*7, 3*12])
                dict_cdf_params[model]['params']['m'] = params1[0]
                dict_cdf_params[model]['params']['s'] = params1[1]
    return dict_cdf_params

def get_html_plots_EOL(EOL,Data):
    # Define the directory path
    directory = "Plots/EOL"

    # Check if directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    for model in EOL.keys():
        if (len(EOL[model]['params'])!=0)&(Data[(Data.model==model)&(Data.status=='ok')].shape[0]>=1):
            max_life = 12*15
            if EOL[model]['distribution'] == 'weibull':
                lamda = EOL[model]['params']['l']
                k = EOL[model]['params']['k']
                y = weibull_cdf(range(0,max_life), k, lamda)
            if EOL[model]['distribution'] == 'normal':
                m = EOL[model]['params']['m']
                s = EOL[model]['params']['s']
                y = normal_cdf(range(0,max_life), m, s)

            line_plot = go.Scatter(x=list(range(max_life)), y=y*100, mode='lines')


            # Combine the plots
            fig = go.Figure(data=[line_plot])

            # Update the layout (optional)
            fig.update_layout(xaxis_title="Life (months)", yaxis_title="EoL Probability (%)")

            fig.write_html("Plots/EOL/"+model+"_EOL_curve.html")
    return 0

def forecast_failures(SM_data, months_ahead,dict_EOL_params):
    dict_output_model_failure = {}
    for model in SM_data.model.unique():
        data2 = SM_data[SM_data.model == model]
        dist_curv = dict_EOL_params[model]['distribution']
        N = dict_EOL_params[model]['N']
        if not dist_curv:
            dict_output_model_failure[model] = {'Expected Failures': 0}
        else:
            A = data2['life_duration'].value_counts()
            if dist_curv == 'weibull':
                l = dict_EOL_params[model]['params']['l']
                k = dict_EOL_params[model]['params']['k']
                probs = weibull_cdf(A.index.values.astype('float') + months_ahead, k, l) - weibull_cdf(
                    A.index.values.astype('float'), k, l)
            if dist_curv == 'normal':
                m = dict_EOL_params[model]['params']['m']
                s = dict_EOL_params[model]['params']['s']
                probs = normal_cdf(A.index.values.astype('float') + months_ahead, m, s) - normal_cdf(
                    A.index.values.astype('float'), m, s)

            Ni = (np.multiply(probs, A.values).sum() * N / A.sum()).round().astype(int)
            dict_output_model_failure[model] = {'Expected Failures': Ni}
    return dict_output_model_failure

def create_population_estimations(data, EOL, months):
    Failure_data  = forecast_failures(data, months,EOL)
    Expected_Failures = pd.DataFrame(Failure_data).transpose()
    Expected_Failures.sort_values('Expected Failures',ascending=False,inplace=True)
    ids_show = (Expected_Failures>=1).sum()[0]
    return Expected_Failures[0:ids_show]

def generate_curves(df_historics):
    Meter_data = prepare_input_for_EOL_curve(df_historics)
    dict_cdf_params = EOL_curve_fit(Meter_data)
    get_html_plots_EOL(dict_cdf_params,Meter_data)
    return Meter_data, dict_cdf_params

def train_xgboost_model(DATA):
    DATA.Replacement_Date = pd.DatetimeIndex(DATA.Replacement_Date)
    id = (DATA["Replacement_Date"] >= datetime(year=2024, month=1, day=1))
    DATA2 = DATA.loc[~id]
    Y = DATA2.loc[:, 'status'] == 'Fault'
    flag = 1
    columns_drop = ['status','life','Replacement_Date','point','sSB','life_duration','brand','model']
    X = pd.DataFrame(DATA2.values,columns=DATA2.columns,index=DATA2.index)
    while flag>=0.5:
        X = X.drop(columns=columns_drop)
        for col in X.columns:
            X[col] = X[col].astype('float')
        X.dropna(inplace=True)
        Y = Y[X.index]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3,stratify=Y)
        bst = XGBClassifier(n_estimators=15, max_depth=2, learning_rate=0.4, objective='binary:logistic', enable_categorical = True,
                                            scale_pos_weight = 1*sum(Y==0) / sum(Y==1))
        bst.fit(X_train, y_train)
        columns_drop = bst.feature_names_in_[bst.feature_importances_<0.05]
        if len(columns_drop)>=1:
            flag=1
        else:
            flag=0
    # Define the directory path
    directory = "Model/"

    # Check if directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
    # After fitting the model
    dump(bst, 'Model/xgboost_model.joblib')
    return 0


def make_predictions(DATA,dict_cdf_params):
    bst = load('Model/xgboost_model.joblib')
    DATA.Replacement_Date = pd.DatetimeIndex(DATA.Replacement_Date)
    id = (DATA["Replacement_Date"] >= datetime(year=2024, month=1, day=1))
    DATA2 = DATA.loc[~id]

    Xpred = DATA2[DATA2.status == 'ok']
    Xpred = Xpred[bst.feature_names_in_.tolist()]

    Probs = pd.DataFrame(bst.predict_proba(Xpred)[:, 1], columns=['Failure Probability'])
    Probs.loc[:, 'model'] = DATA2[DATA2.status == 'ok'].model.values
    Probs.loc[:, 'point'] = DATA2[DATA2.status == 'ok'].point.values
    Probs.loc[:, 'Substation'] = DATA2[DATA2.status == 'ok'].sSB.values
    Probs.index = Probs['point']
    Probs.drop(columns=['point'], inplace=True)

    for i in Probs.index:
        model = Probs.loc[i, 'model']
        life = DATA2.loc[DATA2.point == i, 'life_duration'].values[0]
        if dict_cdf_params[model]['distribution']:
            if dict_cdf_params[model]['distribution'] == 'Normal':
                m = dict_cdf_params[model]['params']['m']
                s = dict_cdf_params[model]['params']['s']
                pf = normal_cdf(life + 1, m, s)
            if dict_cdf_params[model]['distribution'] == 'weibull':
                l = dict_cdf_params[model]['params']['l']
                k = dict_cdf_params[model]['params']['k']
                pf = weibull_cdf(life + 1, k, l)
        else:
            pf = 0.01
        cf = 1 if pf >= 0.1 else 0.5
        Probs.loc[i, 'Failure Probability'] = 100*Probs.loc[i, 'Failure Probability'] * cf
        Probs = Probs.sort_values(by=['Failure Probability'], ascending=[False])
    return Probs





