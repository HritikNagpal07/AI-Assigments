from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import uvicorn

app = FastAPI()

# Load fine-tuned model
base_model = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, "./fine-tuned-model")

class Query(BaseModel):
    ingredients: str

def generate_recipe(ingredients: str) -> str:
    prompt = f"### User: What recipe can I make with {ingredients}?\n### Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=200, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    return response.split("### Assistant:")[-1].strip()

@app.post("/recipe")
async def get_recipe(query: Query):
    try:
        recipe = generate_recipe(query.ingredients)
        return {"ingredients": query.ingredients, "recipe": recipe}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
