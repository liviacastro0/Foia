import os
import requests
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

# Configurações
MODEL_PATH = os.path.join(os.getcwd(), 'modelo_soil.pkl')
DATA_PATH = os.path.join(os.getcwd(), 'data_soil.csv')
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "gpt-4o-mini"  # modelo disponível na nuvem

# Carregar modelo e dados
modelo = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
df.columns = ["nitrogenio", "fosforo", "potassio", "temperatura", "umidade", "ph", "chuva", "cultura"]

def converter_float(valor_str):
    try:
        return float(valor_str.replace(',', '.'))
    except:
        return 0.0

def recomendar_para_web(nitrogenio, fosforo, potassio, temperatura, umidade, ph, chuva):
    input_data = pd.DataFrame([[nitrogenio, fosforo, potassio, temperatura, umidade, ph, chuva]], 
                              columns=["nitrogenio", "fosforo", "potassio", "temperatura", "umidade", "ph", "chuva"])
    cultura_recomendada = modelo.predict(input_data)[0]
    df_filtrado = df[df["cultura"] != cultura_recomendada]
    distancias = euclidean_distances(df_filtrado.drop(columns=["cultura"]), input_data)
    indice_mais_proximo = distancias.flatten().argmin()
    cultura_alternativa = df_filtrado.iloc[indice_mais_proximo]["cultura"]
    return cultura_recomendada, cultura_alternativa

def gerar_dicas_openrouter(cultura):
    prompt = (
       f"Você é um especialista em agricultura. Forneça dicas curtas e diretas sobre como cultivar {cultura}. "
       f"Use até 3 linhas para cada tópico. Não repita informações e evite termos vagos.\n"
       f"1. Preparação do solo: como preparar o solo para o cultivo de {cultura}.\n"
       f"2. Adubação: qual o melhor tipo de adubo e quando aplicar para {cultura}.\n"
       f"3. Controle de pragas: como prevenir e controlar pragas no cultivo de {cultura}.\n"
       f"4. Irrigação: qual o tipo e a frequência de irrigação recomendados para {cultura}.\n"
       f"5. Melhor época de plantio: quando é a melhor época para plantar {cultura}.\n"
       f"6. Outras práticas recomendadas: dicas extras para melhorar o cultivo de {cultura}.\n\n"
       f"Responda de forma clara, objetiva e prática, sem usar termos vagos ou redundantes."
    )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        data = response.json()
        # Extrair texto da resposta do modelo
        resposta_texto = data['choices'][0]['message']['content']
        return resposta_texto
    except Exception as e:
        return f"Erro ao gerar dicas: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resposta', methods=['GET'])
def resposta():
    dados = {
        "nitrogenio": float(request.args.get('nitrogenio', 0)),
        "fosforo": float(request.args.get('fosforo', 0)),
        "potassio": float(request.args.get('potassio', 0)),
        "temperatura": float(request.args.get('temperatura', 0)),
        "umidade": converter_float(request.args.get('umidade', '0')),
        "ph": converter_float(request.args.get('ph', '0')),
        "chuva": converter_float(request.args.get('chuva', '0'))
    }
    
    recomendada, alternativa = recomendar_para_web(**dados)
    dicas_recomendada = gerar_dicas_openrouter(recomendada)
    dicas_alternativa = gerar_dicas_openrouter(alternativa)

    return render_template('resposta.html',
                           **dados,
                           cultura=recomendada,
                           alternativa=alternativa,
                           dicas_recomendada=dicas_recomendada,
                           dicas_alternativa=dicas_alternativa)

if __name__ == '__main__':
    app.run(debug=True)

