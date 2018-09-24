import numpy as np

from func1D_incognito import func1D_incognito as funcao
from func1D_incognito import grad_func1D_incognito as grad_funcao

import matplotlib.pyplot as plt

def gradiente_descida(x_inicio, f, grad_funcao):
# Precisao da solucao
    precisao = 0.001

# Baixa taxa de aprendizado
    taxa_aprendizado = 0.01

# limite de interacoes
    max_inter = 10000
    x_novo = x_inicio
    res = []
    for i in range(max_inter):
        x_velho = x_novo

        x_novo = x_velho - taxa_aprendizado * grad_funcao(x_velho)
        f_x_novo = funcao(x_novo)
        f_x_velho = funcao(x_velho)
        res.append([x_novo, f_x_novo])

# print(f_x_novo - f_x_velho)

        if(abs (f_x_novo - f_x_velho) < precisao):
            print("Precisao alcancada: %f " % (f_x_novo - f_x_velho))
            return np.array(res)
    print("Iteracao maxima alcancada")
    return np.array(res)



# visualizacao da funcao

def main_plot():
    valores = []
    inicio_intervalo = 0
    fim_intervalo = 8
    passo = 0.01
    intervalo = np.arange(inicio_intervalo, fim_intervalo, passo)

    for x in (intervalo):
        y = funcao(x)
        valores.append(y)    
# Forma da funcao no intervalo dado 
    plt.plot(intervalo, list(valores))    


main_plot()
plt.show()

# Descida de gradiente a partir de diferentes valores inciais

for j in range(9):
    minimos = []
    x_inicio = j
    retorno = gradiente_descida(x_inicio, funcao, grad_funcao)
    print("Minimo local em: %f " % retorno[-1][0])
    x_inicio = j
    retorno = gradiente_descida(x_inicio, funcao, grad_funcao)
# Valores da funcao
    plt.plot(retorno[:,0], retorno[:, 1], '+')
    main_plot()
    plt.show()



