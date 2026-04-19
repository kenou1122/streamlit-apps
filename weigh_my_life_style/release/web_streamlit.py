import streamlit as st
import numpy as np
import pandas as pd
import os
from WLS import *

st.write(os.getcwd())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

df_raw = pd.read_csv('./weigh_my_life_style/release/df_example.csv')
model = MLP()
model.load_state_dict(torch.load('model_released.pth'))

######

TITLE = "Weigh my :rainbow[*Life Style*]"

st.set_page_config(
    page_title="Weigh my Life Style",
    page_icon= "🔮"
)

hcol1,hcol2= st.columns([4,2],gap="small",vertical_alignment="bottom")

with hcol1:
    st.title(TITLE)
    
with hcol2:
    egg = st.button(':color[by 雨林\n]{foreground="hsl(210,50%,40%)" background="hsl(210,100%,90%)"}',type="tertiary")

st.caption("*根据生活方式预测明天的体重变化*")

######

pred = ['?','?'] # [tomarrow weight, delta weight]

col1,col2,col3,col4 = st.columns([1.8,1.1,1,2],gap="small",vertical_alignment="center")
with col1:
    st.write("填写以下 3 步信息, 然后点击 ")
with col2:
    submit = st.button('🔮 开始预测')
with col3:
    st.write(" 吧!")
with col4:
    placeholder = st.empty()
    placeholder.metric(label="预计明日体重", 
                       value=pred[0], 
                       delta=pred[1],delta_color="inverse")

######

tab1, tab2, tab3 = st.tabs(["Step 1. 身高体重", "Step 2. 运动生活", "Step 3. 饮食记录"])

with tab1:
    df_raw.loc[0,'Age'] = st.slider(
        "我的**年龄**是",18,68,help='未成年人 ( 或长者 ) 可选择 18 ( 或 68 )',
        key='_q_age',format="%d 岁",value=None
    )
    
    _choice_gender = st.radio(
        "我的**性别**是",['🤵‍♂️ 男','🤵‍♀️ 女','🦹 非二元'],
        key='_q_gender'
    )
    df_raw.loc[0,'Gender'] = {'🤵‍♂️ 男':'Male', '🤵‍♀️ 女':'Female', '🦹 非二元':'Other'}[_choice_gender]
    
    df_raw.loc[0,'Height_cm'] = st.slider(
        "我的**身高**大约是",120,220,help='厘米 cm',
        key='_q_height',format="%d cm"
    )
    
    df_raw.loc[0,'Initial_Weight_kg'] = st.slider(
        "我此时的**体重**大约是",40,140,help="千克 kg",
        key='_q_weight',format="%d kg"
    )

with tab2:
    _choice_worktype = st.selectbox(
        "我进行的**运动类型**是",['🏋️‍♂️ 力量训练','🧘‍♀️ 瑜伽','⛹️‍♀️ 有氧运动','🥱 没有运动'],
        key='_q_worktype'
    )
    df_raw.loc[0,'Workout_Type'] = {'🏋️‍♂️ 力量训练':'Strength', '🧘‍♀️ 瑜伽':'Yoga',
                                 '⛹️‍♀️ 有氧运动':'Cardio', '🥱 没有运动':'None'}[_choice_worktype]
    
    df_raw.loc[0,'Workout_Intensity'] = st.slider(
        "我的**运动强度**是",1,10,help='1 表示轻度活动, 10 表示高强度锻炼',
        key='_q_workintense'
    )
    df_raw.loc[0,'Steps'] = st.slider(
        "我的**走路步数**是",0,20000,help='若大于 20,000 步可直接选择最大值',
        key='_q_steps',step=100,format="%d 步"
    )
    df_raw.loc[0,'Stress_Level'] = st.slider(
        "我感到**生活压力**是",1,10,help='1 表示没有压力, 10 表示压力山大',
        key='_q_stress'
    )
    df_raw.loc[0,'Sleep_Hours'] = st.slider(
        "我的**睡眠时长**是",1,12,help='小时',
        key='_q_sleep',format="%d 小时"
    )
    df_raw.loc[0,'Temp_C'] = st.slider(
        "最近的**气温**是",-10,50,help='摄氏度',
        key='_q_temp',format="%d 摄氏度"
    )

with tab3:
    df_raw.loc[0,'Caffeine_mg'] = st.slider(
        "我摄入了多少**咖啡因** ☕️",0,740,help='一杯 350 mL 的美式咖啡含 200 mg 咖啡因',
        key='_q_caff',step=10, format="%d 毫克"
    )
    df_raw.loc[0,'Calories_Consumed'] = st.slider(
        "我摄入了多少**卡路里** 🧁",0,5000,help='100 克蔬菜小于 40 大卡, 米面小于 200 大卡, 可乐 ( 一瓶 500 ml 约 215 大卡 ), 巧克力 ( 586 大卡 ), 蛋糕 ( 348 大卡 ), 炸薯条 ( 375 大卡 ), 方便面 ( 473 大卡 )',
        key='_q_calo',step=100, format="%d 大卡"
    )
    df_raw.loc[0,'Protein_g'] = st.slider(
        "我摄入了多少**蛋白质** 🥩",0,300,#help='',
        key='_q_protein',step=10, format="%d 克"
    )
    df_raw.loc[0,'Fat_g'] = st.slider(
        "我摄入了多少**脂肪** 🍟",0,160,#help='',
        key='_q_fat',step=10, format="%d 克"
    )
    df_raw.loc[0,'Carbs_g'] = st.slider(
        "我摄入了多少**碳水化合物** 🍚",0,550,#help='',
        key='_q_carbs',step=10, format="%d 克"
    )

#submit = st.button('🔮 开始预测')

if submit:
    
    df_clean = clean_df_raw(df_raw)
    dataset = LWDataset(df_clean.loc[[0]])
    pred[1] = model(dataset.x).item()
    pred[0] = df_raw.loc[0,'Initial_Weight_kg'] + pred[1]
    #st.write(pred[1])
    with col4:
        #placeholder = st.empty()
        placeholder.metric(label="预计明日体重", 
                           value=pred[0], format="%.2f kg",
                           delta=pred[1],delta_color="inverse")

if egg: st.balloons()
# Training Data: https://www.kaggle.com/datasets/waddahali/lifestyle-weight-tracker
