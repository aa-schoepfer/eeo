#Title: Enzyme Expression Optimization
#Author: Alexandre Schoepfer
#Version: 06.02.2021

#Import libraries
import math
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from io import StringIO

#Non-generalized generation of test samples
def test_samples():
 
    ss = [] #samples, list
 
    for x in np.arange(18,25,1):
        for y in np.arange(16,76,4):
            for z in np.arange(3.5,7.25,0.25):
                ss.append([x,y,z])
 
    return np.asarray(ss)

#Generalized range of samples by finding minima and maxima of dataset and applying steps.
def range_samples(df,feats,steps):
 
    ranges = []
 
    for i,f in enumerate(feats):
        ranges.append( np.arange( np.min(df[feats[i]]), np.max(df[feats[i]]) + steps[i], steps[i] ) )  

    return np.array(np.meshgrid(*ranges)).T.reshape(-1,len(feats))

#Generalized range of samples, out of range values
def search_samples(df,feats,steps):

    ranges = []

    for i,f in enumerate(feats):
        ranges.append( np.arange( np.min(df[feats[i]]) - steps[i] , np.max(df[feats[i]]) + 2 * steps[i], steps[i] ) )    
 
    return np.array(np.meshgrid(*ranges)).T.reshape(-1,len(feats))

#Checks if same elements are found in specified array, return binary array of maximum.
#This is to avoid predictions, which are out of range by more than one factor.
@st.cache
def compare_samples(arr1, arr2):
    #arr1 is compared to arr2
    maxarr = []

    for x in arr1:
        temarr = []
        for y in arr2:
            if x.size == y.size:
                temarr.append( (x==y).sum() )
            else:
                raise Exception("Array must have same size.")       
        maxarr.append(max(temarr))

    return maxarr

#Fit and return Gaussian Process model from data
def get_model(df, feats, label, StSc):
        
        X = df[feats].copy()
        y = df[label].copy()

        X = X.to_numpy()
        y = y.to_numpy()

        X = StSc.transform(X)
        y = y.reshape(-1,1)

        model = GaussianProcessRegressor()
        model.fit(X,y)

        return model

#Heatmap generator
def draw_heatmap(*args, **kwargs):

    data = kwargs.pop('data')

    d = data.pivot(args[1], args[0], args[2]).fillna(0)
    
    g = sns.heatmap(d, **kwargs,vmin=0, vmax=args[3], cbar_kws={'label': "Expression ("+args[2]+" )"}, cmap=sns.cm.rocket_r) #, annot=True,annot_kws={'fontsize':4})
    g.invert_yaxis()

#Heatmap grid settings
def grid_plot(d,factors,label):
    
    grid = sns.FacetGrid(d, col=factors[0], col_wrap=4)
    grid.map_dataframe(draw_heatmap, factors[1], factors[2], label, np.max(d[label]))
    grid.set_axis_labels(factors[1], factors[2])
    grid.set_titles(col_template=factors[0]+" {col_name}", row_template="{row_name}")

    st.pyplot(grid)

st.title("Enzyme Expression Optimization Tool")
st.header("Experimental Data")
uploaded_file = st.file_uploader("Upload File",type=['txt','csv'])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    df_cols = list(df.columns)
    df_defs = list(df.columns[0:3])

    featsel = st.multiselect('Features', df_cols, default=df_defs) #Selection of features
  
    df_diff = list(df.copy().drop(columns=featsel).columns)
    df_ddef = list(df_diff[0:1])
    labesel = st.multiselect('Label(s)', df_diff, default=df_ddef) #Selection of labels

    optfor = st.radio("Optimize for", labesel) #Label to be optimized
    oor = False #Out of range

    if st.button("Plot data"):
        with st.spinner(text='plotting...'):
            
            if len(featsel) < 3:
                st.error("Please select at least three features.")
            elif not optfor:
                st.error("Please enter at least one label.")
            else:

                try:

                    dp = df.groupby(featsel[0:3]).median() #Chose median if mutiple results with same features are found.
                    dn = dp.to_csv()
                    dc = pd.read_csv(StringIO(dn)) #Pandas won't do what I want, so I have to come up with this kind of stuff...

                    if np.max(dc[optfor]) <= 1.0: #Check if label is in percents
                        maxv = 1
                    else:
                        maxv = np.max(dc[optfor])

                    grid_plot(dc,featsel,optfor)

                except KeyError:
                    
                    dc = df

                    if np.max(dc[optfor]) <= 1.0:
                        maxv = 1
                    else:
                        maxv = np.max(dc[optfor])

                    grid_plot(dc,featsel,optfor)

    st.header('Prediction')
    st.subheader('Settings')
    if featsel and labesel:
        steps = []
        def_steps = [1.0,4.0,0.25]

        for ind,fs in enumerate(featsel):
            if len(featsel) == 3:
                steps.append( st.number_input(fs+" steps", value=def_steps[ind]) )
            else:
                steps.append( st.number_input(fs+" steps", value=1.00) )

        forSc = search_samples(df,featsel,steps).copy()

        if 2**16 < len(forSc):
            st.warning("The sample range is high. This could take a long time.")

        StSc = StandardScaler()
        StSc.fit(forSc) #Fit on widest range
        
        predi = st.radio("Prediction of", ['Best','Custom'])

        if predi == 'Best':
            oor = st.checkbox("Predict out of range (slower, beta)")
        
        elif predi == 'Custom':
            custom = []
            def_custom = [22.0,48.0,7.00]

            for ind,fs in enumerate(featsel):
                if len(featsel) == 3:
                    custom.append( st.number_input(fs, value=def_custom[ind]) )
                else:
                    custom.append( st.number_input(fs, value=1.00) )
        
        else:
            st.error("Please chose a prediction option.")

        if st.button('Predict'):
            
            if predi == 'Best':
                
                mmodel = get_model(df, featsel, optfor, StSc)
                
                if oor:
                    st.info("For out of range predictions, check the prediction plots to see if the results make sense.")
                    srb = compare_samples(search_samples(df,featsel,steps).copy(), range_samples(df,featsel,steps).copy())
                    test = StSc.transform(search_samples(df,featsel,steps).copy())
                    pred = mmodel.predict(test)
                    for i,p in enumerate(pred):
                        if srb[i] < max(srb)-1:
                            pred[i] = 0
                    best = pred.argmax()
                else:
                    test = StSc.transform(range_samples(df,featsel,steps).copy())
                    pred = mmodel.predict(test)
                    best = pred.argmax()

                st.write( "Best condition:")
                bc = StSc.inverse_transform(test[best])
                st.dataframe(pd.DataFrame({ f:[bc[i]] for i,f in enumerate(featsel)}))

                for l in labesel:
                    lmodel = get_model(df, featsel, l, StSc)
                    st.write("Expected {0}: ".format(l) + '{:.2f}'.format( lmodel.predict([test[best]])[0][0] ) )

            elif predi == 'Custom':
                
                if custom:
                    for l in labesel:                    
                        lmodel = get_model(df, featsel, l, StSc)
                        st.write("Expected {0}: ".format(l) + '{:.2f}'.format( lmodel.predict(StSc.transform([custom]))[0][0] ) )

                else:
                    st.error("Please fill in the custom settings.")

            else:
                st.error("Please chose a prediction option.")

        if len(featsel) == 3:
            st.subheader('Profile')

            plotsel = st.multiselect('Axes', featsel, default=featsel)

            if st.button("Plot predictions"):
                with st.spinner(text='plotting...'):
                    
                    if len(plotsel) != 3:
                        st.error("Please select exactly three features.")
                    elif not optfor:
                        st.error("Please enter at least one label.")
                    else:
                        
                        if oor:
                            d_test_samp = search_samples(df,featsel,steps).copy()
                        else:
                            d_test_samp = range_samples(df,featsel,steps).copy()

                        ts = pd.DataFrame(d_test_samp, columns=plotsel)
                        ts[optfor] = np.nan

                        model = get_model(df, featsel, optfor, StSc)
                        
                        for x in ts.iterrows():
                            res, sdev = model.predict(StSc.transform([[ x[1][y] for y in plotsel]]), return_std=True)
                            x[1][optfor] = res[0]

                        if np.max(df[optfor]) <= 1.0:
                            maxv = 1
                        else:
                            maxv = np.max(df[optfor])

                        grid_plot(ts,plotsel,optfor)

        st.subheader('Suggestions')
        
        af = st.radio("Acquisition Function", ['POI','EI','UBC'])

        disp_num = st.number_input("Number of suggestions", value=10)
        
        if st.button("Get Suggestions"): #Bayesian optimization
        
            #st.info("It is not recommended to test conditions which are out of range by more than two factors.")

            def surrogate(X, model):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    return model.predict(X,return_std=True)       

            def acquisition(Xt, X, model, af):
                
                mut, _ = surrogate(Xt,model)
                mu,std = surrogate(X,model)
                
                MiMa = MinMaxScaler()

                mut = MiMa.fit_transform(mut.reshape(-1,1))
                mu = MiMa.transform(mu.reshape(-1,1))
                
                xi = 0.00001 #Change if needed
                kappa = 2.5 #This too 

                std = std.reshape(-1,1)
                mso = np.max(mut)

                if af == 'POI': #Probability of improvement
                    probs = norm.cdf((mu - mso - xi)/std)
                
                elif af == 'EI': #Expected improvement
                    imp = mu - mso - xi
                    Z = imp/std
                    Z[std == 0] = 0
                    probs = imp * norm.cdf(Z) + std * norm.pdf(Z)
                
                elif af == 'UBC': #Upper bound conjecture
                    probs = mu + kappa * std

                else:
                    probs = None
                    st.error("Aqcusition function not valid.")

                return probs


            X = df[featsel].copy()
            y = df[optfor].copy()

            X = X.to_numpy()
            y = y.to_numpy()
            
            X = StSc.transform(X)

            model = GaussianProcessRegressor()
            model.fit(X,y)

            Xt = search_samples(df,featsel,steps).copy()
            Xt = StSc.transform(Xt)

            scores = acquisition(X, Xt, model, af)
            sc = np.argsort(scores, axis=0)[::-1]

            Xt_s = np.array([StSc.inverse_transform(Xt[x])[0] for x in sc])
            Xt_s = pd.DataFrame({ f:Xt_s[:,i] for i,f in enumerate(featsel)})

            st.dataframe(Xt_s[0:disp_num])