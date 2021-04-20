from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import datetime
import os
from prediction_service import prediction
from sklearn.preprocessing import LabelEncoder

webapp_dir = "webapp"
static_dir = os.path.join(webapp_dir, "static")
templates_dir = os.path.join(webapp_dir, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=templates_dir)

# Loading pickle files
model = pickle.load(open("webapp/lgb_model_v2.pkl", "rb"))

channelGrouping_pkl = pickle.load(open('webapp/Column Pickle/channelGrouping.pkl', 'rb'))
device_browser_pkl = pickle.load(open('webapp/Column Pickle/device_browser.pkl', 'rb'))
device_deviceCategory_pkl = pickle.load(open("webapp/Column Pickle/device_deviceCategory.pkl", 'rb'))
device_isMobile_pkl = pickle.load(open('webapp/Column Pickle/device_isMobile.pkl', 'rb'))
device_operatingSystem_pkl = pickle.load(open('webapp/Column Pickle/device_operatingSystem.pkl', 'rb'))
geoNetwork_continent_pkl = pickle.load(open('webapp/Column Pickle/geoNetwork_continent.pkl', 'rb'))
geoNetwork_country_pkl = pickle.load(open('webapp/Column Pickle/geoNetwork_country.pkl', 'rb'))
geoNetwork_city_pkl = pickle.load(open('webapp/Column Pickle/geoNetwork_city.pkl', 'rb'))
geoNetwork_subContinent_pkl = pickle.load(open('webapp/Column Pickle/geoNetwork_subContinent.pkl', 'rb'))
geoNetwork_metro_pkl = pickle.load(open('webapp/Column Pickle/geoNetwork_metro.pkl', 'rb'))
geoNetwork_networkDomain_pkl = pickle.load(open('webapp/Column Pickle/geoNetwork_networkDomain.pkl', 'rb'))
geoNetwork_region_pkl = pickle.load(open('webapp/Column Pickle/geoNetwork_region.pkl', 'rb'))
trafficSource_adContent_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_adContent.pkl', 'rb'))
trafficSource_adwordsClickInfo_page_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_adwordsClickInfo_page.pkl', 'rb'))
trafficSource_adwordsClickInfo_gclId_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_adwordsClickInfo_gclId.pkl', 'rb'))
trafficSource_adwordsClickInfo_slot_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_adwordsClickInfo_slot.pkl', 'rb'))
trafficSource_campaign_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_campaign.pkl', 'rb'))
trafficSource_isTrueDirect_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_isTrueDirect.pkl', 'rb'))
trafficSource_keyword_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_keyword.pkl', 'rb'))
trafficSource_medium_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_medium.pkl', 'rb'))
trafficSource_referralPath_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_referralPath.pkl', 'rb'))
trafficSource_source_pkl = pickle.load(open('webapp/Column Pickle/trafficSource_source.pkl', 'rb'))


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/batch", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        try:

            path = request.files['file']
            path = path.filename
            response = prediction.form_response(path)
            result = "The Top Ten Predictions are "
            return render_template("index1.html", response=[response], result=result)


        except Exception as e:
            error = {"error": e}
            return render_template("404.html", error=error)

    else:
        return render_template('index1.html')


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    # Extracting the weekday, month, year, visitHour
    date = request.form["Date"]
    month = pd.to_datetime(date, format="%Y-%m-%dT%H:%M").month
    year = pd.to_datetime(date, format="%Y-%m-%dT%H:%M").year
    weekday = pd.to_datetime(date, format="%Y-%m-%dT%H:%M").weekday


    visitHour = int(pd.to_datetime(date, format='%Y-%m-%dT%H:%M').hour)

    # Extracting visitNumber
    visitNumber = float(request.form["visitNumber"])

    #  Extracting totals.hits
    totals_hits = float(request.form["hits"])

    # Extracting totals.pageviews
    totals_pageviews = int(request.form["pageviews"])

    # Extracting totals.newVisits
    totals_newVisits = 1#request.form["newVisits"]

    # Extracting channelGrouping
    channelGrouping = str(request.form['channelGrouping'])

    # Extracting browser 
    device_browser = str(request.form['browser'])

    # Extracting operatingSystem
    device_operatingSystem = str(request.form['operatingSystem'])

    # Extracting isMobile
    device_isMobile = str(request.form['isMobile'])

    # Extracting deviceCategory
    device_deviceCategory = str(request.form['deviceCategory'])

    # Extracting continent
    geoNetwork_continent = str(request.form['continent'])

    # Extracting subContinent
    geoNetwork_subContinent = str(request.form['subContinent'])

    # Extracting country
    geoNetwork_country = str(request.form['myCountry'])

    # Extracting region
    geoNetwork_region = str(request.form['myregion'])

    # Extracting metro
    geoNetwork_metro = str(request.form['mymetro'])

    # Extracting city
    geoNetwork_city = str(request.form['mycity'])

    # Extracting network Domain
    geoNetwork_networkDomain = str(request.form['mynetworkDomain'])

    trafficSource_campaign = str(request.form['campaign'])

    trafficSource_source = str(request.form['trafficsource'])

    trafficSource_medium = str(request.form['medium'])

    trafficSource_keyword = str(request.form['traffickeyword'])

    trafficSource_isTrueDirect = str(request.form['isTrueDirect'])

    trafficSource_referralPath = str(request.form['trafficreferral'])

    trafficSource_adwordsClickInfo_page = str(request.form['adwordsClickInfopage'])

    trafficSource_adwordsClickInfo_slot = str(request.form['adwordsClickInfoslot'])

    trafficSource_adwordsClickInfo_gclId = str(request.form['trafficgclid'])

    trafficSource_adContent = str(request.form['adContent'])

    data_for_df = [{'channelGrouping': channelGrouping, 'visitNumber': visitNumber, 'device_browser': device_browser,
                    'device_operatingSystem': device_operatingSystem,
                    'device_isMobile': device_isMobile, 'device_deviceCategory': device_deviceCategory,
                    'geoNetwork_continent': geoNetwork_continent,
                    'geoNetwork_subContinent': geoNetwork_subContinent, 'geoNetwork_country': geoNetwork_country,
                    'geoNetwork_region': geoNetwork_region, 'geoNetwork_metro': geoNetwork_metro,
                    'geoNetwork_city': geoNetwork_city, 'geoNetwork_networkDomain': geoNetwork_networkDomain,
                    'totals_hits': totals_hits,
                    'totals_pageviews': totals_pageviews, 'totals_newVisits': totals_newVisits,
                    'trafficSource_campaign': trafficSource_campaign, 'trafficSource_source': trafficSource_source,
                    'trafficSource_medium': trafficSource_medium,
                    'trafficSource_keyword': trafficSource_keyword,
                    'trafficSource_isTrueDirect': trafficSource_isTrueDirect,
                    'trafficSource_referralPath': trafficSource_referralPath,
                    'trafficSource_adwordsClickInfo_page': trafficSource_adwordsClickInfo_page,
                    'trafficSource_adwordsClickInfo_slot': trafficSource_adwordsClickInfo_slot,
                    'trafficSource_adwordsClickInfo_gclId': trafficSource_adwordsClickInfo_gclId,
                    'trafficSource_adContent': trafficSource_adContent, 'weekday': weekday,
                    'month': month, 'year': year, 'visitHour': visitHour}]

    df = pd.DataFrame(data_for_df)


    df['channelGrouping'] = channelGrouping_pkl.transform(df['channelGrouping'])
    df['device_browser']= device_browser_pkl.transform(df['device_browser'])
    df['device_deviceCategory'] = device_deviceCategory_pkl.transform(df['device_deviceCategory'])
    df['device_isMobile']= device_isMobile_pkl.transform(df['device_isMobile'])
    df['device_operatingSystem'] = device_operatingSystem_pkl.transform(df['device_operatingSystem'])
    df['geoNetwork_city'] = geoNetwork_city_pkl.transform(df['geoNetwork_city'])
    df['geoNetwork_continent'] = geoNetwork_continent_pkl.transform(df['geoNetwork_continent'])
    df['geoNetwork_country'] = geoNetwork_country_pkl.transform(df['geoNetwork_country'])
    df['geoNetwork_metro'] = geoNetwork_metro_pkl.transform(df['geoNetwork_metro'])
    df['geoNetwork_networkDomain'] = geoNetwork_networkDomain_pkl.transform(df['geoNetwork_networkDomain'])
    df['geoNetwork_region'] = geoNetwork_region_pkl.transform(df['geoNetwork_region'])
    df['geoNetwork_subContinent'] = geoNetwork_subContinent_pkl.transform(df['geoNetwork_subContinent'])
    df['trafficSource_adContent']= trafficSource_adContent_pkl.transform(df['trafficSource_adContent'])
    df['trafficSource_adwordsClickInfo_gclId'] = trafficSource_adwordsClickInfo_gclId_pkl.transform(df['trafficSource_adwordsClickInfo_gclId'])
    df['trafficSource_adwordsClickInfo_page'] = trafficSource_adwordsClickInfo_page_pkl.transform(df['trafficSource_adwordsClickInfo_page'])
    df['trafficSource_adwordsClickInfo_slot'] = trafficSource_adwordsClickInfo_slot_pkl.transform(df['trafficSource_adwordsClickInfo_slot'])
    df['trafficSource_source'] = trafficSource_source_pkl.transform(df['trafficSource_source'])
    df['trafficSource_campaign'] = trafficSource_campaign_pkl.transform(df['trafficSource_campaign'])
    df['trafficSource_isTrueDirect'] = trafficSource_isTrueDirect_pkl.transform(df['trafficSource_isTrueDirect'])
    df['trafficSource_keyword'] = trafficSource_keyword_pkl.transform(df['trafficSource_keyword'])
    df['trafficSource_medium'] = trafficSource_medium_pkl.transform(df['trafficSource_medium'])
    df['trafficSource_referralPath'] = trafficSource_referralPath_pkl.transform(df['trafficSource_referralPath'])




    # Making predictions
    prediction= model.predict(df)
    output = round(prediction[0], 4)
    if output < 0:
        return render_template('home.html', prediction_text="Transaction Revenue is less than 0")
    else:
        return render_template('home.html', prediction_text="Transaction Revenue is {} $".format(output))


if __name__ == "__main__":
    app.run(debug=True)
