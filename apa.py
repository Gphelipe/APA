import numpy as np
import random
from collections import deque, Counter
import pandas as pd

def carregar_matriz_C(caminho_csv):
    df = pd.read_csv(caminho_csv)  # Carrega o arquivo CSV

    # Extrair todos os pares únicos de animais cruzados
    todos_pares = list(set(df['Animal_1']).union(set(df['Animal_2'])))

    # extração de machos e femeas
    machos = sorted({p.split('_')[0] for p in todos_pares})   
    femeas = sorted({p.split('_')[1] for p in todos_pares})   

    NF = len(femeas)  # Número de fêmeas
    NM = len(machos)  # Número de machos

    tamanho = NF * NM   # Total de possíveis cruzamentos
    C = np.zeros((tamanho, tamanho))  # Inicializa a matriz de coancestralidade

    # mapeamento: macho_femea → índice
    par_para_indice = {f"{m}_{f}": i for i, (f, m) in enumerate([(f, m) for f in femeas for m in machos])}

    # Preenche a matriz de coancestralidade
    for _, linha in df.iterrows():
        a1, a2, coef = linha['Animal_1'], linha['Animal_2'], linha['Coef']

        if a1 in par_para_indice and a2 in par_para_indice:
            i = par_para_indice[a1]
            j = par_para_indice[a2]
            C[i, j] = coef
            C[j, i] = coef  # Simetria da matriz

    # Adiciona um ruído pequeno para evitar zeros na matriz e melhorar a busca 
    C += np.random.uniform(0, 0.001, size=C.shape)

    return C, femeas, machos, par_para_indice


def avaliar(P, C, NF, NM):
    #Calcula o custo total de coancestralidade de uma solução P (lista de machos atribuídos a cada fêmea),
        # somando C[idx1, idx2] para todos os pares de cruzamentos.
    custo = 0.0
    for i in range(NF):      #vai percorer todas combinações das femeas e acumular os coeficientes entre os decendentes 
        for j in range(NF):
            idx1 = i * NM + P[i]    # Índice do cruzamento da fêmea i com macho P[i]
            idx2 = j * NM + P[j]    # Índice do cruzamento da fêmea j com macho P[j]
            custo += C[idx1, idx2]  # Soma dos coeficientes entre os cruzamentos
    return custo

def gerar_vizinhos(P, NM, max_uso):
    #Trocas simples: mudar um macho por outro.
    #Trocas duplas: trocar os machos atribuídos entre duas fêmeas.
    
    vizinhos = []
    NF = len(P)
    uso = Counter(P)     # Conta quantas vezes cada macho está sendo usado
    
    # Vizinhos por troca simples (Gera novas soluções substituindo o macho de uma fêmea por outro macho que ainda não atingiu o limite de uso)
    for i in range(NF):
        for m in range(NM):
            if m != P[i] and uso[m] < max_uso[m]:
                nova = P.copy()
                nova[i] = m
                vizinhos.append(nova)
    
    # Vizinhos vizinhos trocando os machos entre duas fêmeas diferentes.
    for i in range(NF):
        for j in range(i+1, NF):
            if P[i] != P[j]:
                nova = P.copy()
                nova[i], nova[j] = nova[j], nova[i]
                vizinhos.append(nova)
    
    return vizinhos

def solucao_inicial(NF, NM, max_uso):
    #Cria uma solução inicial válida, distribuindo os machos entre as fêmeas, respeitando max_uso.

    P = []                  #Prepara os machos embaralhados e uma contagem de uso para distribuir equitativamente.
    machos_disponiveis = list(range(NM))
    random.shuffle(machos_disponiveis)
    
    # Distribui os machos para cada fêmea respeitando o limite de uso
    contagem = Counter()
    for _ in range(NF):
        for m in machos_disponiveis:
            if contagem[m] < max_uso[m]:
                P.append(m)
                contagem[m] += 1
                break
        else:
            # Se não encontrou, pega qualquer macho disponível
            P.append(random.choice(machos_disponiveis))
    
    return P

def busca_tabu(C, NF, NM, max_uso, iter_max=1000, tabu_tam=20):
    #Mantém uma lista tabu de soluções recentes.
    #Gera vizinhos e escolhe o melhor que não está na tabu.
    #Atualiza a melhor solução global.
    #Exibe progresso a cada 100 iterações.

    P = solucao_inicial(NF, NM, max_uso)  #Gera uma solução inicial viável e avalia seu custo.
    melhor_P = P.copy()
    melhor_custo = avaliar(P, C, NF, NM)
    
    lista_tabu = deque(maxlen=tabu_tam)     # Inicia a lista tabu com a solução atual.
    lista_tabu.append(tuple(P))
    
    for it in range(iter_max):                      #cada iteração, gera e embaralha os vizinhos da solução atual.
        vizinhos = gerar_vizinhos(P, NM, max_uso)
        random.shuffle(vizinhos)  # Explorar mais diversidade
        
        melhor_vizinho = None                           #Seleciona o vizinho de menor custo que não esteja na lista tabu.
        melhor_custo_vizinho = float('inf')
        
        for vizinho in vizinhos:
            if tuple(vizinho) not in lista_tabu:
                custo = avaliar(vizinho, C, NF, NM)
                if custo < melhor_custo_vizinho:
                    melhor_vizinho = vizinho
                    melhor_custo_vizinho = custo
        
        if melhor_vizinho is None:
            break  # Encerra se nenhum vizinho novo foi encontrado

           # Atualiza a solução atual e, se for melhor que a anterior, atualiza o melhor global.  
        P = melhor_vizinho
        lista_tabu.append(tuple(P))
        
        if melhor_custo_vizinho < melhor_custo:
            melhor_P = P.copy()
            melhor_custo = melhor_custo_vizinho
        
        if it % 100 == 0:
            print(f"[{it}] Custo atual: {melhor_custo_vizinho:.4f} | Melhor: {melhor_custo:.4f}")
    
    return melhor_P, melhor_custo

if __name__ == "__main__":
    caminho_csv = "parentesco_produtos.csv"
    C, femeas, machos, par_para_indice = carregar_matriz_C(caminho_csv)
    NF = len(femeas)
    NM = len(machos)

    print(f"Número de fêmeas: {NF}, Número de machos: {NM}")
    print(f"Dimensão da matriz C: {C.shape}")
    print(f"Valores não-zero na matriz C: {np.count_nonzero(C)}")

    # Ajuste o número máximo de usos por macho conforme necessário
    max_uso = {i: 2 for i in range(NM)}  # Cada macho pode ser usado até 2 vezes

    # Executar a busca tabu
    solucao, custo = busca_tabu(C, NF, NM, max_uso, iter_max=1000)

    # Exibir resultados
    print("\nCruzamentos finais:")
    for i, macho_idx in enumerate(solucao):
        print(f" {machos[macho_idx]} × {femeas[i]}")

    print(f"\nCusto total de coancestralidade: {custo:.4f}")