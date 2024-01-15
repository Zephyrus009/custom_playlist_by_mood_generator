import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
import pandas as pd
import requests
import json
import os

load_dotenv()

# Accessing IP-based location information
ip_url = "https://api.ipify.org?format=json"
response = requests.get(ip_url)
ip_data = json.loads(response.text)
ip_address = ip_data["ip"]

# Using a free weather API 
weather_api_key = os.environ['weather_api_key'] 
weather_api_base_url = "http://api.openweathermap.org/data/2.5/weather"

# Retrieving location information based on IP address
geolocation_url = f"https://freegeoip.app/json/{ip_address}"
geolocation_response = requests.get(geolocation_url)
geolocation_data = json.loads(geolocation_response.text)
city = geolocation_data.get("city")

# Making the weather API request
weather_url = f"{weather_api_base_url}?q={city}&appid={weather_api_key}"
weather_response = requests.get(weather_url)
weather_data = json.loads(weather_response.text)

# Extracting relevant weather information
if weather_data.get("cod") == 200:  # Check for successful response
    weather_description = weather_data["weather"][0]["description"]
    temperature_kelvin = weather_data["main"]["temp"]
    temperature_celsius = temperature_kelvin - 273.15  # Convert to Celsius
    feels_like = round(weather_data["main"]["feels_like"] - 273.15, 2)
    humidity = weather_data["main"]["humidity"]
    wind_speed = weather_data["wind"]["speed"]
    weather_details = f"It's {city} and here weather condition is {weather_description} and temperature is {temperature_celsius:.2f}°C"
    print(weather_details)
else:
    print("Error: Unable to retrieve weather data.")

st.header(f"**Weather in {city} is {weather_description} and now its {datetime.now().strftime('%I:%M %p')}**")
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", f"{temperature_celsius:.2f}°C")
col2.metric("Wind", f"{wind_speed}mph")
col3.metric("Humidity", f"{humidity}%")

## Genarating Moods

llm = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human = True,temperature=0)
moods_data = pd.read_csv("moods_data.csv")
relative_moods = list(moods_data['Moods'])

def song_mood_analyzer(weather_details,relative_moods):
    template1 = """Can you suggest me specific song moods by the weather condition and temperature, 
            {weather_details}, I want only two song moods and not the song name, any kind of description is also not required, 
            return only headings one after another with a comma between them, output should be in a python list mood1,mood2 """
    prompt1 = PromptTemplate(input_variables=["weather_details"],template=template1)
    chain1 = LLMChain(llm=llm,prompt=prompt1,output_key="moods")

    template2 = """
                    find out the similar and strongly specific mood from the list of moods {relative_moods},
                    which are similar with the list {moods}, calculate the similarity score and also see on basis of similarity score, 
                    i want only mood headings only,  no bullet points, I want only similar moods in a pattern like mood1|mood2|mood3...
                """
    prompt2 = PromptTemplate(input_variables=["moods","relative_moods"],template=template2)
    chain2 = LLMChain(llm=llm,prompt=prompt2,output_key="founded_moods")

    final_chain = SequentialChain(
    chains=[chain1,chain2],
    input_variables=["weather_details","relative_moods"],
    output_variables=["moods","founded_moods"],
    verbose=True
    )

    result = final_chain({"weather_details":f"{weather_details}","relative_moods":f"{relative_moods}"})

    return result

founded_moods = song_mood_analyzer(weather_details,relative_moods)["founded_moods"]
founded_moods = founded_moods.split("|")

updated_playlist = moods_data[moods_data["Moods"].isin(founded_moods)].reset_index(drop=True)
updated_playlist = updated_playlist[["artist","song"]]

st.header("Let the Weather Sing the Soundtrack: Matching Melodies to Mother Nature")
st.dataframe(updated_playlist,hide_index=True)