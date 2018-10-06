import numpy as np
from func2D_incognito_NEW import func2D_incognito as funcao
from func2D_incognito_NEW import grad_func2D_incognito as grad_funcao
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#Funcao descida de gradiente para duas variaveis
def gradiente_descida(inicio, f, grad_funcao):
# Precisao da solucao
    precisao = 0.001

# Baixa taxa de aprendizado
    taxa_aprendizado = 0.01

 # limite de interacoes
    max_inter = 10000
    x_novo = inicio
    res = []
    for i in range(max_inter):
        x_velho = x_novo

        x_novo = x_velho - taxa_aprendizado * grad_funcao(x_velho)
        f_x_novo = funcao(x_novo)
        f_x_velho = funcao(x_velho)
        res.append([x_novo, f_x_novo])

        print(f_x_novo - f_x_velho)
#Valors de minimos locais
        if(abs (f_x_novo - f_x_velho) < precisao):
            print("Precisao alcancada: %f " % (f_x_novo - f_x_velho))
            return np.array(res)
    print("Iteracao maxima alcancada")
    return np.array(res)

#intervalo para amostragem
inicio_intervalo = [-3,-3]
fim_intervalo = [3,3]
passo = 0.1
 
lista1 = []
lista2 = []
lista3 = []

#valores para calculo da funcao
for var1, var2 in zip(np.arange(inicio_intervalo[0], fim_intervalo[0], passo), np.arange(inicio_intervalo[1], fim_intervalo[1], passo)):

    inicio = [var1, var2]
    retorno = gradiente_descida(inicio, funcao, grad_funcao)
    print("Minimo local em: ")
    print(retorno[-1][0])
    lista1.append(var1)
    lista2.append(var2)
    
var1 = np.array(lista1)
var2 = np.array(lista2)    

#Geracao do grafico com o comportamento da funcao no intervalo considerado

eixo_X, eixo_Y = np.meshgrid(var1, var2)

eixo_X_flatten = eixo_X.reshape(-1)
eixo_Y_flatten = eixo_Y.reshape(-1)

valores_funcao = np.array([funcao([i, j]) for i, j in zip(eixo_X_flatten, eixo_Y_flatten)])
valores_funcao_shape = valores_funcao.reshape(eixo_X.shape)

eixo_Z = valores_funcao_shape
graph = plt.axes(projection='3d')
graph.set_xlabel('x Values')
graph.set_ylabel('y Values')
graph.set_zlabel('z(x,y) values')
graph.plot_surface(eixo_X, eixo_Y, eixo_Z, color='green')
plt.show()
