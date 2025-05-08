import os
from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import ollama
import markdown  # Necessário para converter markdown em HTML

app = Flask(__name__)

# Configurações
MODEL_PATH = os.path.join(os.getcwd(), 'modelo_soil.pkl')
DATA_PATH = os.path.join(os.getcwd(), 'data_soil.csv')
OLLAMA_MODEL = "gemma:2b"

# Carregar o modelo e os dados
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

def gerar_dicas_ollama(cultura):
    prompt = (
        f"Dicas rápidas para cultivar {cultura}.\n"
        f"Responda em português e use markdown.\n"
        f"Separe cada item com uma linha em branco. Use no máximo 3 linhas por item.\n\n"
        f"1. *Preparação do solo*: Descreva brevemente como deve ser preparado o solo para essa cultura.\n"
        f"2. *Adubação*: Informe qual tipo de fertilizante é recomendado.\n"
        f"3. *Controle de pragas*: Sugira maneiras práticas de controlar pragas.\n"
        f"4. *Irrigação*: Indique a quantidade e frequência de irrigação.\n"
        f"5. *Melhor época de plantio*: Detalhe a melhor época do ano para plantar.\n"
        f"6. *Outras práticas recomendadas*: Forneça dicas adicionais relevantes.\n\n"
        f"Use uma linguagem direta, objetiva e clara. Respostas concisas, sem repetições."
    )

    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={'temperature': 0.5, 'num_ctx': 1024}
        )
        if 'response' not in response:
            return "⚠️ Não foi possível obter dicas no momento. Tente novamente mais tarde."
        # Converte markdown para HTML
        return markdown.markdown(response['response'])
    except Exception as e:
        return f"⚠️ Erro ao gerar dicas: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/criarconta')
def criar_conta():
    return render_template('criarconta.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

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

    dicas_recomendada = gerar_dicas_ollama(recomendada)
    dicas_alternativa = gerar_dicas_ollama(alternativa)

    return render_template('resposta.html',
                           **dados,
                           cultura=recomendada,
                           alternativa=alternativa,
                           dicas_recomendada=dicas_recomendada,
                           dicas_alternativa=dicas_alternativa)

@app.route('/processar_login', methods=['POST'])
def processar_login():
    email = request.form.get('email')
    senha = request.form.get('senha')
    return redirect(url_for('chat'))

if __name__ == '__main__':
    try:
        ollama.list()
        print("Conexão com Ollama estabelecida com sucesso!")
    except Exception as e:
        print(f"Erro ao conectar com Ollama: {e}")
        print("Certifique-se que o Ollama está instalado e rodando localmente")
    
    app.run(debug=True)
