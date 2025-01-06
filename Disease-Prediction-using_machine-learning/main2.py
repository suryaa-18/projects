import streamlit as st
import numpy as np
import pandas as pd
import sqlite3

conn = sqlite3.connect('diseaseinfo.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS DiseaseData (id INTEGER PRIMARY KEY, name TEXT, age TEXT, symptom1 TEXT, symptom2 TEXT, symptom3 TEXT, symptom4 TEXT, symptom5 TEXT, disease TEXT)''')

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria',
'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
'distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum',
'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads',
'scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails',
'blister','red_sore_around_nose','yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
'Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox',
'Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis',
'Tuberculosis','Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)','Heartattack','Varicoseveins',
'Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis','Arthritis',
'(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis','Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,
'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,
'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,
'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,'(vertigo) Paroymsal  Positional Vertigo':36,
'Acne':37,'Urinary tract infection':38,'Psoriasis':39,'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

def treatment(i):
    if i==0:
        st.write("Eat Yoghurt, apply coconut oil or tea tree oil. If it starts spreading consult a dermatologist")
    if i==1:
        st.write("Antihistamines can help relieve sneezing, itching, a stuffy or runny nose, and watery eyes. Examples of oral antihistamines include cetirizine (Zyrtec Allergy), fexofenadine (Allegra Allergy) and loratadine (Claritin, Alavert).")
    if i==2:
        st.write("Maintain a healthy weight. Elevate the head of your bed. Avoid foods and drinks that trigger reflux. Eat food slowly and chew thoroughly.")
    if i==3:
        st.write("Fill half of your plate with fruits and vegetables for every meal: Fresh, canned, frozen and dried fruits and vegetables are all good options. Antibiotics.")
    if i==4:
        st.write("Antihistamines. Your provider may prescribe an antihistamine or recommend a nonprescription antihistamine such as diphenhydramine (Benadryl). An antihistamine can block immune system chemicals triggered during an allergic reaction.")
    if i==5:
        st.write("Most stomach ulcers are caused by an infection and will require antibiotics, along with other medications. The sooner you start antibiotics and other treatments, the sooner your ulcer will heal.")
    if i==6:
        st.write("HIV is treated with antiretroviral medicines, which work by stopping the virus replicating in the body. This allows the immune system to repair itself and prevent further damage. A combination of HIV drugs is used because HIV can quickly adapt and become resistant.")
    if i==7:
        st.write("Regular physical activity can help control blood sugar levels and reduce the risk of developing type 2 diabetes. A healthy diet that includes fruits, vegetables, whole grains, lean proteins, and low-fat dairy products can help manage diabetes.")
    if i==8:
        st.write("Gastroenteritis get better on their own without medical treatment. You can treat viral gastroenteritis by replacing lost fluids and electrolytes to prevent dehydration. In some cases, over-the-counter medicines may help relieve your symptoms.")
    if i==9:
        st.write("Quick-relief inhalers (bronchodilators) quickly open swollen airways that are limiting breathing. In some cases, allergy medications are necessary. Long-term asthma control medications, generally taken daily, are the cornerstone of asthma treatment.")
    if i==10:
        st.write("Balance nutrients. Go for less sodium (under 1,500 mg per day) and more potassium. Put probiotics on your side. Eating food that contains probiotics—consumable live bacteria—has been linked to healthier blood pressure.")
    if i==11:
        st.write("Try a Cold Pack. If you have a migraine, place a cold pack on your forehead. Use a Heating Pad or Hot Compress. Ease Pressure on Your Scalp or Head.")
    if i==12:
        st.write("Avoid sitting in the same position for an extended period. Avoid sleeping on a mattress that is too soft. Avoid sleeping where an air-conditioner or a fan blows directly at you.")
    if i==13:
        st.write("Hemorrhagic Stroke Treatment. This may involve medications to lower blood pressure and prevent seizures, as well as surgical interventions to repair damaged blood vessels or remove blood clots. In some cases, a ventriculostomy may be performed to drain excess fluid from the brain and relieve pressure.")
    if i==14:
        st.write("If a blocked bile duct is to blame, your doctor may suggest surgery to open it. If your skin is itching, your doctor can prescribe cholestyramine to be taken by mouth. This medication is used to remove bile acids from your body, which cause itching.")
    if i==15:
        st.write("The preferred antimalarial for interim oral treatment is artemether-lumefantrine (Coartem®) because of its fast onset of action. Other oral options include atovaquone-proguanil (Malarone™), quinine, and mefloquine. IV or oral clindamycin and tetracyclines, such as doxycycline, are not adequate for interim treatment.")
    if i==16:
        st.write("Resting. Drinking plenty of fluids to prevent dehydration. using paracetamol to bring down fevers. using creams or lotions, such as calamine lotion, to reduce the itching – if you have a skin condition such as eczema ask your doctor or pharmacist about other available creams.")
    if i==17:
        st.write("There is no specific treatment for dengue. The focus is on treating pain symptoms. Acetaminophen (paracetamol) is often used to control pain. Non-steroidal anti-inflammatory drugs like ibuprofen and aspirin are avoided as they can increase the risk of bleeding.")
    if i==18:
        st.write("Fluoroquinolones. These antibiotics, including ciprofloxacin (Cipro), may be a first choice. Cephalosporins. This group of antibiotics keeps bacteria from building cell walls.")
    if i==19:
        st.write("There is no specific treatment for hepatitis A. Recovery from symptoms following infection may be slow and can take several weeks or months. It is important to avoid unnecessary medications that can adversely affect the liver, e.g. acetaminophen, paracetamol.")
    if i==20:
        st.write("Antiviral medications. Several antiviral medicines — including entecavir (Baraclude), tenofovir (Viread), lamivudine (Epivir), adefovir (Hepsera) and telbivudine — can help fight the virus and slow its ability to damage your liver.")
    if i==21:
        st.write("Antiviral medications, including sofosbuvir and daclatasvir, are used to treat hepatitis C. Some people's immune system can fight the infection on their own and new infections do not always need treatment. Treatment is always needed for chronic hepatitis C.")
    if i==22:
        st.write("Treat chronic hepatitis D with medicines called interferons, such as peginterferon alfa-2a link (Pegasys). Researchers are studying new treatments for hepatitis D. In addition, medicines for hepatitis B may be needed.")
    if i==23:
        st.write("There is no specific treatment capable of altering the course of acute hepatitis E. As the disease is usually self-limiting, hospitalization is generally not required. It is important to avoid unnecessary medications that can adversely affect liver function, e.g. acetaminophen, paracetamol.")
    if i==24:
        st.write("Corticosteroids. These medicines might help some people with severe alcoholic hepatitis live longer. However, corticosteroids have serious side effects. They're not likely to be used if you have failing kidneys, stomach bleeding or an infection.")
    if i==25:
        st.write("If you have an active TB disease you will probably be treated with a combination of antibacterial medications for a period of six to 12 months. The most common treatment for active TB is isoniazid INH in combination with three other drugs—rifampin, pyrazinamide and ethambutol.")
    if i==26:
        st.write("There's no cure for the common cold. Most cases of the common cold get better without treatment within 7 to 10 days. But a cough may last a few more days.")
    if i==27:
        st.write("Pneumonia can be serious so its important to get treatment quickly. The main treatment for bacterial pneumonia is antibiotics. You should also rest and drink plenty of water. If you are diagnosed with bacterial pneumonia, your doctor should give you antibiotics to take within four hours.")
    if i==28:
        st.write("With sclerotherapy, your health care provider injects a chemical solution into the hemorrhoid tissue to shrink it. While the injection causes little or no pain, it might be less effective than rubber band ligation. Coagulation. Coagulation techniques use laser or infrared light or heat.")
    if i==29:
        st.write("Hospitals use techniques to restore blood flow to the part of the heart muscle damaged during your heart attack: You might receive clot-dissolving drugs (thrombolysis), balloon angioplasty, surgery or a combination of treatments")
    if i==30:
        st.write("Sclerotherapy works best for spider veins. These are small varicose veins. Laser treatment can be used on the surface of the skin. Small bursts of light can make small varicose veins disappear.Phlebectomy treats surface varicose veins. ")
    if i==31:
        st.write("An underactive thyroid (hypothyroidism) is usually treated by taking daily hormone replacement tablets called levothyroxine. Levothyroxine replaces the thyroxine hormone, which your thyroid does not make enough of.")
    if i==32:
        st.write("Anti-thyroid medicine. These medications slowly ease symptoms of hyperthyroidism by preventing the thyroid gland from making too many hormones. Anti-thyroid medications include methimazole and propylthiouracil.")
    if i==33:
        st.write("Glucagon is the first-line and only approved treatment for severe hypoglycaemia in a person out of the hospital with impaired consciousness who is unable to administer fast-acting carbohydrates orally.")
    if i==34:
        st.write("For OA in general, the most helpful advice is to maintain an ideal weight, avoid overusing joints that are damaged and follow a plan of exercise that strengthens the muscles supporting the joint. Your doctor or physical therapist should be able to help you with any of these.")
    if i==35:
        st.write("Healthy eating and arthritis. Most people find that they feel better if they eat a balanced and varied diet to get all the vitamins, minerals, antioxidants and other nutrients their body needs. Try to eat a Mediterranean-style diet which includes fish, pulses, nuts, olive oil and plenty of fruit and vegetables.")
    if i==36:
        st.write("The most effective benign paroxysmal positional vertigo treatments involve physical therapy exercises. The goal of these exercises is to move the calcium carbonate particles out of your semicircular canals and back into your utricle. Here, the particles resorb more easily and don't cause uncomfortable symptoms.")
    if i==37:
        st.write("Benzoyl peroxide works as an antiseptic to reduce the number of bacteria on the surface of the skin. It also helps to reduce the number of whiteheads and blackheads, and has an anti-inflammatory effect. Benzoyl peroxide is usually available as a cream or gel. It's used either once or twice a day.")
    if i==38:
        st.write("Taking antibiotics, prescribed by a healthcare professional, at home can treat most UTIs. However, some cases may require treatment in a hospital.")
    if i==39:
        st.write("Steroid creams or ointments (topical corticosteroids) are commonly used to treat mild to moderate psoriasis in most areas of the body. The treatment works by reducing inflammation. This slows the production of skin cells and reduces itching. Topical corticosteroids range in strength from mild to very strong.")
    if i==40:
        st.write("A doctor might recommend a topical ointment for only a few sores. Oral antibiotics can be used when there are more sores. Use the prescription exactly as the doctor says to. Once the sores heal, someone with impetigo is usually not able to spread the bacteria to others.")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming your data is in a DataFrame format
def preprocess_data(df):
    # Drop any rows with missing values
    df.dropna(inplace=True)

    # Assuming the target variable is in the column 'target'
    X = df.drop(columns=['target'])  # Features
    y = df['target']  # Target variable

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # Impute missing values for numerical features
    numerical_cols = X.select_dtypes(include=['int', 'float']).columns
    imputer = SimpleImputer(strategy='mean')
    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, y_train, y_test

# Define your DecisionTree function
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split 
from sklearn import tree
import numpy as np

def calcAcc(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1score = f1_score(y_test, y_pred, average='weighted')

    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1score)

# X_train, X_test, y_train, y_test = preprocess_data(df)

def DecisionTree(X, y, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, l1, l2, disease):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree Classifier
    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = clf3.predict(X_test)
    calcAcc(y_test, y_pred)

    psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]

    for k in range(len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    for a in range(len(disease)):
        if predicted == a:
            pred_disease = disease[a]
            return pred_disease
            
    else:
        return "Disease not found"

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

# Define your randomforest function
def randomforest(X, y, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, l1, l2, disease):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X_train, np.ravel(y_train))

    # Calculate accuracy
    y_pred = clf4.predict(X_test)
    calcAcc(y_test, y_pred)

    psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]

    for k in range(len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted = predict[0]

    for a in range(len(disease)):
        if predicted == a:
            return disease[a]
            break
    else:
        return "Disease not found"

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

# NaiveBayes function
def NaiveBayes(X, y, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, l1, l2, disease):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Naive Bayes Classifier
    gnb = GaussianNB()
    gnb = gnb.fit(X_train, np.ravel(y_train))

    # Calculate accuracy
    y_pred = gnb.predict(X_test)
    calcAcc(y_test, y_pred)

    psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]

    for k in range(len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted = predict[0]

    for a in range(len(disease)):
        if predicted == a:
            return disease[a], a
            
    else:
        return "Disease not found"

def sqlconnection(PatName, PatAge, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, pdisease):
    c.execute("INSERT INTO DiseaseData (name, age, symptom1, symptom2, symptom3, symptom4, symptom5, disease) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (PatName, PatAge, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, pdisease))
    conn.commit()
    st.success('Data inserted successfully!')
def patdata():
    st.subheader('Existing Data of all patients:')
    result = c.execute("SELECT * FROM DiseaseData").fetchall()
    for row in result:
            id = row[0]  # Assuming the first column is the attribute name
            st.write("PATIENT - ",id)
            name = row[1]  # Assuming the first column is the attribute name
            age = row[2]  # Assuming the second column is the attribute value
            s1 = row[3]  # Assuming the first column is the attribute name
            s2 = row[4]  # Assuming the second column is the attribute value
            s3 = row[5]  # Assuming the first column is the attribute name
            s4 = row[6]  # Assuming the second column is the attribute value
            s5 = row[7]  # Assuming the first column is the attribute name
            pdis = row[8]  # Assuming the second column is the attribute value
            st.write(f"Patient name - {name}")
            st.write(f"Patient age - {age}")
            st.write(f"Symptom 1 - {s1}")
            st.write(f"Symptom 2 - {s2}")
            st.write(f"Symptom 3 - {s3}")
            st.write(f"Symptom 4 - {s4}")
            st.write(f"Symptom 5 - {s5}")
            st.write(f"Predicted disease - {pdis}")
            st.write("")

def getpatdata():
    st.subheader('Data of the given patient:')
    # Retrieve all data names from the table
    # Retrieve all data names from the table
    all_data = [row[0] for row in c.execute("SELECT name FROM DiseaseData").fetchall()]

        # Select multiple data items to display
    selected_data1 = st.multiselect('Select data to display:', all_data)

    if selected_data1:
        query = "SELECT * FROM DiseaseData WHERE name IN ({seq})".format(seq=','.join(['?']*len(selected_data1)))
        selected_rows = c.execute(query, selected_data1).fetchall()
        for row in selected_rows:
            name = row[1]  # Assuming the first column is the attribute name
            age = row[2]  # Assuming the second column is the attribute value
            s1 = row[3]  # Assuming the first column is the attribute name
            s2 = row[4]  # Assuming the second column is the attribute value
            s3 = row[5]  # Assuming the first column is the attribute name
            s4 = row[6]  # Assuming the second column is the attribute value
            s5 = row[7]  # Assuming the first column is the attribute name
            pdis = row[8]  # Assuming the second column is the attribute value
            st.write(f"Patient name - {name}")
            st.write(f"Patient age - {age}")
            st.write(f"Symptom 1 - {s1}")
            st.write(f"Symptom 2 - {s2}")
            st.write(f"Symptom 3 - {s3}")
            st.write(f"Symptom 4 - {s4}")
            st.write(f"Symptom 5 - {s5}")
            st.write(f"Predicted disease - {pdis}")

def max_occurrence(words):
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    max_word = max(word_count, key=word_count.get)
    max_count = word_count[max_word]
    return max_word, max_count

# Streamlit UI
def main():
    st.set_page_config(layout="centered")
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #122e33;
    opacity: 0.8;
    
    background-size: 10px 10px;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title(':blue[Disease Prediction System]')
    st.title('Using Machine Learning algorithms')
    st.title('  ')

    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # st.markdown('<p class="big-font">Please enter the patient details</p>', unsafe_allow_html=True)
    st.subheader("Please enter the patient details")
    cursor = conn.cursor()
    PatName = st.text_input("Patient name", placeholder="Enter the name..")
    PatAge = st.number_input("Patient age", min_value=0, max_value=120, step=1, value=0, format="%d", help="Enter the age..")

    if len(PatName) < 2:
        st.error("Name should be at least 2 characters long.")
    # col1, col2, col3 = st.columns([1, 1, 1])
    # with col3:

    st.title('  ')
    st.title('  ')
    st.subheader("Select Symptoms")
    Symptom1 = st.selectbox('Symptom 1', l1, index=None, placeholder="Select symptom")
    Symptom2 = st.selectbox('Symptom 2', l1, index=None, placeholder="Select symptom")
    Symptom3 = st.selectbox('Symptom 3', l1, index=None, placeholder="Select symptom")
    Symptom4 = st.selectbox('Symptom 4', l1, index=None, placeholder="Select symptom")
    Symptom5 = st.selectbox('Symptom 5', l1, index=None, placeholder="Select symptom")
   

    # col1, col2, col3 = st.columns([1, 1, 1])
    # with col2:
    if st.button("Predict the disease"):
        # Pass appropriate data to your DecisionTree function
        pdisease1,a=NaiveBayes(X, y, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, l1, l2, disease)
        pdisease2=DecisionTree(X, y, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, l1, l2, disease)
        pdisease3=randomforest(X, y,  Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, l1, l2, disease)
        n=a
         # Example usage:
        a=pdisease1
        b=pdisease2
        c=pdisease3
        words = [a,b,c]
        max_word, max_count = max_occurrence(words)
        pdisease=max_word
        st.write(pdisease)
        treatment(n)
        sqlconnection(PatName, PatAge, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, pdisease)
        patdata()

    st.write('')
    st.write('')

    st.subheader("Details of all the patients")
    if st.button("Get details"):
        patdata()

    getpatdata()


    st.write('')
    st.write('')

    st.subheader("Delete data")
    cursor = conn.cursor()
    data_names = cursor.execute("SELECT name FROM DiseaseData").fetchall()

    # Extract the names from the fetched data
    data_names = [name[0] for name in data_names]

    # Display the select box with the extracted names
    selected_data = st.selectbox('Select data to delete:', data_names)
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        if st.button("Delete entered patient data"):
            selected_data = selected_data[0] if isinstance(selected_data, tuple) else selected_data
            cursor.execute("DELETE FROM DiseaseData where name=?", (selected_data,))
            conn.commit()
            st.success('Data deleted successfully!')

    with col3:
        if st.button("Delete all data"):
            cursor.execute("DELETE FROM DiseaseData")
            conn.commit()
            st.success('Data deleted successfully!')

if __name__ == "__main__":
    main()