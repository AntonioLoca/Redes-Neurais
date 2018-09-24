import numpy as np

from func2D_incognito import func2D_incognito as funcao
from func2D_incognito import grad_func2D_incognito as grad_funcao

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

        if(abs (f_x_novo - f_x_velho) < precisao):
            print("Precisao alcancada: %f " % (f_x_novo - f_x_velho))
            return np.array(res)
    print("Iteracao maxima alcancada")
    return np.array(res)


inicio_intervalo = [-2.1, 0.99]
fim_intervalo = [2,4]
passo = 0.1
 
v1 = []
v2 = []
v3 = []

for var1, var2 in zip(np.arange(inicio_intervalo[0], fim_intervalo[0], passo), np.arange(inicio_intervalo[1], fim_intervalo[1], passo)):

    inicio = [var1, var2]
    retorno = gradiente_descida(inicio, funcao, grad_funcao)
    print("Minimo local em: ")
    print(retorno[-1][0])
    v1.append(var1)
    v2.append(var2)
    v3.append(funcao([var1, var2]))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = [v1][0]
Y = [v2][0]
Z = [v3][0]

ax.plot(X, Y, Z)
plt.show()
