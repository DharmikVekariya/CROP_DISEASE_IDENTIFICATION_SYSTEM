import streamlit as st

from streamlit_option_menu import option_menu


import tometo, trending, corn, cotton, about
st.set_page_config(
        page_title="Plant Disease",
)



class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Plant Disease ',
                options=['Tometo','Corn','Cotton','About'],
                icons=['t','c','c','c'],
                menu_icon='tree',
                default_index=0,
                styles={
                    "container": {"padding": "5!important","background-color":'black'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},}
                
                )

        
        if app == "Tometo":
            tometo.app()
        if app == "Corn":
            corn.app()    
        if app == "Trending":
            trending.app()        
        if app == 'Cotton':
            cotton.app()
        if app == 'About':
            about.app()    
             
          
             
    run()            
         
