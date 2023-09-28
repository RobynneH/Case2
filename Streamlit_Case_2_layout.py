#!/usr/bin/env python
# coding: utf-8

# # Case 2 Streamlit

# ## Team 4: Sten den Hartog, Robynne Hughes, Wolf Huiberts & Charles Huntington

# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
# import kaggle
import zipfile
import streamlit as st
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# !kaggle datasets download -d teejmahal20/airline-passenger-satisfaction


# In[3]:


# zf = zipfile.ZipFile(r"C:\Users\robyn\OneDrive\Documents\HHS\Minor HvA Data Science\Case 2\airline-passenger-satisfaction.zip") 
# train = pd.read_csv(zf.open("train.csv"), index_col = 0)


# In[4]:


train = pd.read_csv("train.csv", index_col = 0)


# In[5]:


st.set_page_config(layout = "wide")


# In[6]:


st.title("Passenger Satisfaction")


# In[7]:


st.subheader("Team 4: Sten den Hartog, Robynne Hughes, Wolf Huiberts & Charles Huntington")


# In[8]:


image = Image.open('dataset-cover.jpeg')
st.image(image)


# In[9]:


st.write("We will be working with a dataset from Kaggle: ")
st.link_button("Passanger satisfaction", "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")
st.write("Add information about the dataset?")


# **Edit data**

# In[10]:


train.columns = train.columns.str.replace(' ', '_')


# In[11]:


#dropped certain columns from the heatmap to make it easier to read and contain more relevant information.
train_clean = train.drop(columns = ['id', 'Departure/Arrival_time_convenient', 'Gate_location', 
                          'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes'])


# ## Show information data

# In[12]:


st.header("The dataset that we use in this blog")


# In[13]:


head = train.head()
information = train.describe()
describe = train.describe()
nas = train.isna().sum()

dataHead = st.checkbox('Frist 5 rows of the dataframe')
dataInfo = st.checkbox('The information about the dataset')
dataDescribe = st.checkbox("The description of the dataset")
dataNas = st.checkbox("The nonavailable values in the dataset")

if dataHead:
    st.write('First rows of dataframe: ', head)
if dataInfo:
    st.write('Information of the dataset: ')
    st.write(information)
if dataDescribe:
    st.write('Describe the dataset: ')
    st.write(describe)
if dataNas:
    st.write('Nonavailable values in the dataset: ', nas)


# ## Visualisaties van de data

# In[14]:


st.divider()
st.header("The visualization of the data")


# **Pie chart**

# In[15]:


# Define the variables you want to create pie charts for
variables = ['Gender', 'Customer_Type', 'Type_of_Travel', 'Class', 'satisfaction',
             'Inflight_wifi_service', 'Ease_of_Online_booking',
             'Food_and_drink', 'Online_boarding', 'Seat_comfort',
             'Inflight_entertainment', 'On-board_service', 'Leg_room_service',
             'Baggage_handling', 'Checkin_service', 'Inflight_service',
             'Cleanliness']

# Create a dropdown widget for selecting variables
selected_variable = st.selectbox("Select Variable:", variables)

# Create a function to plot pie charts for a selected variable
def plot_pie_chart(selected_variable):
    plt.figure(figsize=(6, 6))

    # Calculate the value counts for the selected variable
    value_counts = train_clean[selected_variable].value_counts()

    # Modify labels for specific variables
    labels = value_counts.index
    if selected_variable == 'Gender':
        labels = ['Male', 'Female']
    elif selected_variable == 'Customer_Type':
        labels = ['Loyal', 'Disloyal']
    elif selected_variable == 'Type_of_Travel':
        labels = ['Personal', 'Business']
    elif selected_variable == 'Class':
        labels = ['Business', 'Eco', 'Eco Plus']
    elif selected_variable == 'satisfaction':
        labels = ['Satisfied', 'Dissatisfied']

    # Create a pie chart
    plt.pie(value_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("YlOrBr"))
    plt.title(f'Distribution of {selected_variable}')
    st.pyplot(plt)

# Call the function to plot the initial pie chart
# plot_pie_chart(selected_variable)

col3, col4 = st.columns(2)

with col3:
    plot_pie_chart(selected_variable)
with col4:
    st.write("Add information about pie chart")


# **Barplot dropdown**

# In[16]:


crosstabGender = pd.crosstab(train.Gender, train.satisfaction).reset_index()
crosstabClass = pd.crosstab(train.Class, train.satisfaction).reset_index()
crosstabCustomer = pd.crosstab(train.Customer_Type, train.satisfaction).reset_index()
crosstabTravel = pd.crosstab(train.Type_of_Travel, train.satisfaction).reset_index()
crosstabAge = pd.crosstab(train.Age, train.satisfaction).reset_index()

fig = go.Figure()
crosstablist = [crosstabGender, crosstabClass, crosstabCustomer,crosstabTravel]

for crosstab in crosstablist:
    column_names = list(crosstab.columns.values)
    fig.add_trace(go.Bar(x = crosstab.iloc[:,0],
                         y = crosstab.iloc[:,1],
                         offsetgroup = 0,
                         name = column_names[1],
                         marker_color = "#e377c2"))
    fig.add_trace(go.Bar(x = crosstab.iloc[:,0], 
                         y = crosstab.iloc[:,2], 
                         offsetgroup = 0, 
                         name = column_names[2],
                         base = crosstab.iloc[:,1],
                         marker_color = "#17becf"))

dropdown_buttons = [{'label':'ALL', 'method':'update','args': [{'visible':[True,True,True,True,True,True,True,True]},
                                                                {'xaxis': {'title': 'All'}}]},
                    {'label':'Gender', 'method':'update', 'args': [{'visible':[True,True,False,False,False,False,False,False]},
                                                                {'xaxis': {'title': 'Gender'}}]},
                    {'label':'Class','method': 'update','args':[{'visible':[False, False, True, True,False,False,False,False]},
                                                               {'xaxis': {'title': 'Class'}}]},
                    {'label':'Type of Customer','method': 'update','args':[{'visible':[False, False, False, False,True,True,False,False]},
                                                               {'xaxis': {'title': 'Type of Customer'}}]},
                    {'label':'Type of Travel','method': 'update','args':[{'visible':[False, False, False, False,False,False,True,True]},
                                                               {'xaxis': {'title': 'Type of Travel'}}]}]

fig.update_layout({'updatemenus':[{'active': 0 , 'buttons':dropdown_buttons}]},yaxis_title = "Count", xaxis_title = "All",
                  title = "Satisfaction of people on airplanes based on variable", legend_title = "Satisfaction")
# st.plotly_chart(fig)

col5, col6 = st.columns(2)

with col5:
    st.plotly_chart(fig)
with col6:
    st.write("Add information about barplots")


# **Heatmap**

# In[17]:


#redefining non-numeric values to numeric for correlation heatmap
train_clean['satisfaction'] = train_clean['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
train_clean['Type_of_Travel'] = train_clean['Type_of_Travel'].map({'Business travel': 1, 'Personal Travel': 0})
train_clean['Class'] = train_clean['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business' : 2})
train_clean['Gender'] = train_clean['Gender'].map({'Male': 1, 'Female': 0})
train_clean['Customer_Type'] = train_clean['Customer_Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})

fig, ax = plt.subplots()
Matrix = train_clean.corr()
sns.heatmap(Matrix[['satisfaction']], annot = True, ax=ax)

# st.pyplot(fig)

col7, col8 = st.columns(2)
with col7:
    st.pyplot(fig)
with col8:
    st.write("Add information about heatmap")


# **Scatter slider**

# In[19]:


fig1 = px.scatter(train, x=train.Departure_Delay_in_Minutes, color=train.satisfaction,color_discrete_map={'satisfied': 'red', 
                                                   'neutral or dissatisfied': 'royalblue'})
fig2 = px.scatter(train, x=train.Arrival_Delay_in_Minutes, color=train.satisfaction, color_discrete_map={'satisfied': 'red', 
                                                   'neutral or dissatisfied': 'royalblue'})

fig1.update_layout(xaxis=dict(range=[train.Departure_Delay_in_Minutes.min(),train.Departure_Delay_in_Minutes.max()+ 10],
                              rangeslider=dict(range=[train.Departure_Delay_in_Minutes.min(),train.Departure_Delay_in_Minutes.max()+10], 
                                               thickness=0.05, bgcolor='#7f7f7f')))

fig2.update_layout(xaxis=dict(range=[train.Arrival_Delay_in_Minutes.min(),train.Arrival_Delay_in_Minutes.max()+ 10],
                              rangeslider=dict(range=[train.Arrival_Delay_in_Minutes.min(),train.Arrival_Delay_in_Minutes.max()+10], 
                                               thickness=0.05, bgcolor='#7f7f7f')))

col9, col10 = st.columns(2)

with col9:
    slider = st.radio("Departure or Arrival", ["Departure", "Arrival"])
    if slider == "Departure":
        st.plotly_chart(fig1)
    else:
        st.plotly_chart(fig2)
with col10:
    st.write("Add information about scatterplot")
    


# In[ ]:


with st.expander("Bronnen"):
    st.write("Bron naar de kaggle website voor de dataset: ", )
    st.write("Bron voor grouped stack bar in een subplot: ", 
             "https://stackoverflow.com/questions/68489063/plotly-how-to-make-group-and-stack-bar-subplots-in-one-figure")
    st.write("Bron naar de rangeslider: ",
            "https://plotly.com/python/range-slider/")

