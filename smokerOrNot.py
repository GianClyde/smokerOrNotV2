from tkinter import *
from ttkbootstrap.constants import *
import ttkbootstrap as tb
from PIL import Image,ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



#functions
def age_scaler(e):
    age_label.config(text=f"{int(age_scale.get())}")

def male_gender():
    result_button["state"] = 'normal'
    accuracy_gauge.stop()
    accuracy_gauge.config(value=0)
    left_col_img_holder.config(image=male_img)
 
    

def female_gender():
    result_button["state"] = 'normal'
    accuracy_gauge.stop()
    accuracy_gauge.config(value=0)
    #result_content.config(text=f"{accuracy_gauge.variable.get()}" )
    left_col_img_holder.config(image=female_img)
    
def logicReg():
    accuracy_gauge.config(value=0)
    result_button["state"] = 'disabled'
    global prob
    df = pd.read_csv("medinsur.csv")

    df['smoker'] = df['smoker'].astype('category')
    df['smoker'] = df['smoker'].cat.codes
    if bmi_entry.get() == '':
        user_bmi = 0
    else:
        user_bmi = float(bmi_entry.get())
    
    if age_scale.get() == '':
        user_age = 0
    else:
        user_age = int(age_scale.get())
        
    if children_entry.get() == '':
        user_children = 0
    else:
        user_children = int(children_entry.get())
        
    
  
   
    if user_gender.get() == "male":

    
        df_male = df[df['sex'] == 'male']
        
        # For "male" subset
        X_male = df_male[["age", "bmi", "children"]]
        y_male = df_male.iloc[:,4].values
        
        X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male,
                                                                        y_male,
                                                                        test_size=0.25,
                                                                        random_state=0)
        model_male = LogisticRegression()
        model_male.fit(X_train_male, y_train_male)
        
        
        coefficients_male = model_male.coef_[0]
        intercept_male = model_male.intercept_[0]
        
        logit_p_male = intercept_male + (coefficients_male[0] * user_age) + (coefficients_male[1] * user_bmi) + \
        (coefficients_male[2] * user_children)

        # Calculate the probability (p) for the male subset using the logistic function
        p_male = 1 / (1 + np.exp(-logit_p_male))
        prob = int(p_male*100)
        prob_display = "{:.2f}".format(p_male*100)
        accuracy_gauge.start()
        
        if prob > 50:
            result_content.config(text="You are a smoker")
        else:
            result_content.config(text="You are not a smoker")
        
     
    
    else:
        df = pd.read_csv("medinsur.csv")

        df['smoker'] = df['smoker'].astype('category')
        df['smoker'] = df['smoker'].cat.codes
        df_female = df[df['sex'] == 'female']
        
        X_female = df_female[["age", "bmi", "children"]]
        y_female = df_female.iloc[:,4].values
        
        X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female,
                                                                                y_female,
                                                                                test_size=0.25,
                                                                                random_state=0)
        
        model_female = LogisticRegression()
        model_female.fit(X_train_female, y_train_female)
        y_pred_female = model_female.predict(X_test_female)
        
        coefficients_female = model_female.coef_[0]
        intercept_female = model_female.intercept_[0]
        
        # Calculate logit(p) for the female subset
        logit_p_female = intercept_female + (coefficients_female[0] * user_age) + (coefficients_female[1] * user_bmi) + \
            (coefficients_female[2] * user_children)

        # Calculate the probability (p) for the female subset using the logistic function
        p_female = 1 / (1 + np.exp(-logit_p_female))
        prob = int(p_female*100)
        prob_display = "{:.2f}".format(p_female*100)
        accuracy_gauge.start()
        if prob > 50:
            result_content.config(text="You are a smoker")
        else:
            result_content.config(text="You are not a smoker")
        

       
           
    
         
root = tb.Window(themename="superhero")

root.title("Smoker or Not")
root.geometry('900x900')

#variables
user_gender =StringVar()

#ui title
ui_header = tb.Label(root, text="Smoker or Not", font=("Arial Black", 20))
ui_header.pack()

#columns
row1 = tb.Frame(root)
left_column = tb.Frame(row1)
right_column = tb.Frame(row1)

left_column.pack(side="left", padx=10)
right_column.pack(side="left",fill='both', expand=True, padx=10)
row1.pack(fill='both', expand=True)


#image
imgM = (Image.open("maleimg.png"))
male_img = ImageTk.PhotoImage(imgM.resize((300,550), Image.ANTIALIAS))
imgF = (Image.open("femaleimg.png"))
female_img = ImageTk.PhotoImage(imgF.resize((300,550), Image.ANTIALIAS))
imgML = (Image.open("maleicon.png"))
male_logo = ImageTk.PhotoImage(imgML.resize((85,80), Image.ANTIALIAS))
imgFL = (Image.open("femaleicon.png"))
female_logo = ImageTk.PhotoImage(imgFL.resize((85,80), Image.ANTIALIAS))

#left column content

left_col_img_holder = tb.Label(left_column, image=male_img)
left_col_img_holder.pack(padx=10)

#right column content
#gender
gender_frame= tb.LabelFrame(right_column, text="Gender",padding=20)
male_button = tb.Radiobutton(gender_frame, variable=user_gender,value="male", text="Male", image=male_logo, bootstyle="info toolbutton, secondary", command=male_gender)
female_button = tb.Radiobutton(gender_frame, variable=user_gender,value="female",text="Female", image=female_logo ,bootstyle="info toolbutton, secondary", command=female_gender)

male_button.pack(side="left",fill='both', expand=True, padx=10)
female_button.pack(side="left",fill='both', expand=True, padx=10)
gender_frame.pack(fill='both',  padx= 30)
#age
age_frame = tb.LabelFrame(right_column, text="Age")
age_scale = tb.Scale(age_frame, bootstyle="warning", length=400,from_=0, to = 150,command=age_scaler)
age_label = tb.Label(age_frame, text= " ", font=("Arial Black", 20))
age_scale.place(relx=0.5, rely=0.4, anchor=CENTER)
age_label.place(relx=0.5, rely=0.7, anchor=CENTER)
age_frame.pack(fill='both', expand=True, padx= 30, anchor=CENTER)

#No. of children
children_frame = tb.LabelFrame(right_column, text="No. of Children")
children_entry = tb.Entry(children_frame)
children_entry.place(relx=0.5, rely=0.5, anchor=CENTER)
children_frame.pack(fill='both', expand=True, padx= 30)

#BMI
bmi_frame = tb.LabelFrame(right_column, text="BMI")
bmi_entry = tb.Entry(bmi_frame)
#BMI layout

bmi_entry.place(relx=0.5, rely=0.5, anchor=CENTER)
bmi_frame.pack(fill='both', expand=True, padx= 30)

#button

result_button = tb.Button(root, text="Show Result",width=50, command=logicReg)
result_button.pack(padx=30, pady=20)

bottom_frame = tb.LabelFrame(root, padding=20, text="Result")
bottom_frame.pack(expand=True, fill='both', pady=20, padx=20)

#bottom frame content
result_content = tb.Label(bottom_frame, text="Result")
result_content.pack(pady=10)
#gauge
accuracy_gauge = tb.Floodgauge(bottom_frame, mask= "Probability: {}%", maximum=100, orient="horizontal", value=0, mode="determinate")
accuracy_gauge.pack(fill='both', padx=40)

def check(*args):
    
    if accuracy_gauge.variable.get() < 25:
        accuracy_gauge.config(bootstyle = 'success')
        if accuracy_gauge.variable.get() == int(prob):
            accuracy_gauge.stop() 
    elif accuracy_gauge.variable.get() > 25 and accuracy_gauge.variable.get() < 50:
        accuracy_gauge.config(bootstyle = 'warning')
        if accuracy_gauge.variable.get() ==  int(prob):
            accuracy_gauge.stop() 
    elif accuracy_gauge.variable.get() > 50:
        accuracy_gauge.config(bootstyle = 'danger')
        if accuracy_gauge.variable.get() ==  int(prob):
            accuracy_gauge.stop() 

        
accuracy_gauge.variable.trace('w', check)

root.mainloop()