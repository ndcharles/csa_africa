# 1. Library imports
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware


class laptop(BaseModel):
    company: str
    typename: str 
    cpu: str
    ram: int
    memory: int
    gpu: str
    os: str
    ppi: float
    
    
#    'Company', 'TypeName', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight',
#       'Price', 'ppi'


# 2. Create app and model objects
app = FastAPI()
model = joblib.load("cat_model.sav")

# 3. Expose the prediction functionality, make a prediction from the passed

@app.get("/")
def read_root():
    return {"Hello": "This is CS Africa Academy 22 ML Project"}

@app.post("/csa/")
def csa(data: laptop):
    st.sidebar.title("RPS Image Classifier")
    data = data.dict()
    company = data['company']
    typename = data['typename']
    cpu = data['cpu']
    ram = data['ram']
    memory = data['memory']
    gpu = data['gpu']
    os = data['os']
    ppi = data['ppi']
    
    result = np.exp(model.predict([company, typename, cpu, ram, memory, gpu, os, ppi]))
    result = np.round(result, 2)
    
    return {'Expected Laptop price for your specifications is': result}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)