from fastapi import FastAPI, Request, Form
import os
import uvicorn
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

@app.get("/")
def form_post(request: Request):
    result = "Type a review"
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})

@app.post("/")
def form_post(request: Request, review_to_predict: str = Form(...)):
    result = os.popen(f"python return_pred.py {review_to_predict}").read()
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})
#    return {
#            "ans": ans
#           }
        
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
